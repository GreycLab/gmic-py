#include "gmicpy.h"

#include <cstdio>
#include <cstdlib>

#include "structmember.h"

using namespace std;

using T = gmic_pixel_type;

//------- G'MIC-PY MACROS ----------//

// Set the GMIC_PY_DEBUG environment variable to any value to enable logging
#ifndef GMIC_PY_LOG
#define GMIC_PY_LOG(msg)           \
    if (getenv("GMIC_PY_DEBUG")) { \
        fprintf(stdout, msg);      \
    }
#endif

//------- G'MIC INTERPRETER INSTANCE BINDING ----------//

static PyObject *
PyGmic_repr(PyGmic *self)
{
    return PyUnicode_FromFormat(
        "<%s interpreter object at %p with _gmic address at %p>",
        Py_TYPE(self)->tp_name, self, self->_gmic);
}

static size_t
get_image_size(const gmic_image<T> *img)
{
    return img->_width * img->_height * img->_depth * img->_spectrum;
}

static size_t
get_image_size(const gmic_image<T> &img)
{
    return get_image_size(&img);
}

/* Copy a GmicImage's contents into a gmic_list at a given position. Run this
 * typically before a gmic.run(). */
static void
swap_gmic_image_into_gmic_list(const PyGmicImage *image, gmic_list<T> &images,
                               const int position)
{
    images[position].assign(
        image->_gmic_image->_width, image->_gmic_image->_height,
        image->_gmic_image->_depth, image->_gmic_image->_spectrum);
    images[position]._width = image->_gmic_image->_width;
    images[position]._height = image->_gmic_image->_height;
    images[position]._depth = image->_gmic_image->_depth;
    images[position]._spectrum = image->_gmic_image->_spectrum;
    memcpy(images[position]._data, image->_gmic_image->_data,
           get_image_size(image->_gmic_image) * sizeof(T));
    images[position]._is_shared = image->_gmic_image->_is_shared;
}

/* Copy a GmicList's image at given index into an external GmicImage. Run this
 * typically after gmic.run(). */
void
swap_gmic_list_item_into_gmic_image(gmic_list<T> &images, const int position,
                                    const PyGmicImage *image)
{
    // Put back the possibly modified reallocated image buffer into the
    // original external GmicImage Back up the image data into the original
    // external image before it gets freed

    swap(image->_gmic_image->_data, images[position]._data);
    image->_gmic_image->_width = images[position]._width;
    image->_gmic_image->_height = images[position]._height;
    image->_gmic_image->_depth = images[position]._depth;
    image->_gmic_image->_spectrum = images[position]._spectrum;
    image->_gmic_image->_is_shared = images[position]._is_shared;

    // Prevent freeing the data buffer's pointer now copied into the external
    // image

    // [MODIF 2022]
    /*
    images[position]._data = 0;
    */
}

#ifdef gmic_py_jupyter_ipython_display

// Cross-platform way to have a temp directory string, through Python
const char *
get_temp_dir()
{
    PyObject *module = NULL;
    PyObject *pystr = NULL;
    module = PyImport_ImportModule("tempfile");
    pystr = PyObject_CallMethod(module, "gettempdir", NULL);
    Py_XDECREF(pystr);
    Py_XDECREF(module);
    return PyUnicode_AsUTF8(pystr);
}

// Cross-platform way to have a unique id string, through Python
const char *
get_uuid()
{
    PyObject *module = NULL;
    PyObject *pystr = NULL;
    module = PyImport_ImportModule("uuid");
    // Using a time-sortable uuid generator for file names
    // See https://stackoverflow.com/a/63970430/420684
    pystr = PyObject_Str(PyObject_CallMethod(module, "uuid1", "II", 0, 0));
    Py_XDECREF(pystr);
    Py_XDECREF(module);
    return PyUnicode_AsUTF8(pystr);
}

// You must free the result if result is non-NULL.
// Modified from https://stackoverflow.com/a/779960/420684
// Returns: a tuple of (adapted_gmic_command_string, ["a list of
// fi*le*","glob*","strings*"])
PyObject *
gmic_py_str_replace_display_to_output(char *orig, char *extension)
{
    PyObject *pyresult = NULL;  // A (result_commands_line, (list,of,filename,
                                // glob,strings)) 2-elements tuple
    PyObject *pyglobs = NULL;   // A (result_commands_line, (list,of,filename,
                                // glob,strings)) 2-elements tuple
    char rep[] = " display";    // The string to seek and replace, ie. needle
    char replacement_command[] =
        " display output ";  // The string to seek and replace, ie. needle
    char *result;            // the return string
    char *ins;               // the next insert point
    char *tmp;               // varies
    int len_rep;             // length of rep (the string to remove)
    int len_with;            // length of with (the string to replace rep with)
    int len_front;           // distance between rep and end of last rep
    int count;               // number of replacements
    char with[512];          // replacement path
    char with_globbed[512];  // replacement path with glob ending

    with[0] = '\0';
    with_globbed[0] = '\0';
    pyglobs = PyList_New(0);
    Py_INCREF(pyglobs);

    // sanity checks and initialization
    if (!orig || !rep) {
        // [MODIF 2022]
        Py_XDECREF(pyglobs);
        return NULL;
    }
    len_rep = strlen(rep);
    if (len_rep == 0) {
        // [MODIF 2022]
        Py_XDECREF(pyglobs);
        return NULL;  // empty rep causes infinite loop during count
    }
    // build a first uuid to detect fixed replacement string length
    // 'with' becomes thus '/tmp/unique-id.png'
    // 'with_globbed' becomes '/tmp/unique-id*.png'
    strcat(with, replacement_command);
    strcat(with, get_temp_dir());
    strcat(with, "/");
    strcat(with, get_uuid());
    strcpy(with_globbed, with);
    strcat(with_globbed, "*");
    strcat(with, extension);
    strcat(with_globbed, extension);
    len_with = strlen(with);

    // count the number of replacements needed
    ins = orig;
    for (count = 0; (tmp = strstr(ins, rep)); ++count) {
        ins = tmp + len_rep;
    }

    tmp = result =
        (char *)malloc(strlen(orig) + (len_with - len_rep) * count + 1);

    if (!result) {
        // [MODIF 2022]
        Py_XDECREF(pyglobs);
        return NULL;  // empty rep causes infinite loop during count
    }

    // first time through the loop, all the variable are set correctly
    // from here on,
    //    tmp points to the end of the result string
    //    ins points to the next occurrence of rep in orig
    //    orig points to the remainder of orig after "end of rep"
    while (count--) {
        PyList_Append(
            pyglobs,
            Py_BuildValue("s", with_globbed + strlen(replacement_command)));
        ins = strstr(orig, rep);

        len_front = ins - orig;
        tmp = strncpy(tmp, orig, len_front) + len_front;
        tmp = strcpy(tmp, with) + len_with;
        orig += len_front + len_rep;  // move to next "end of rep"

        // recompute a uuid file path for each replacement
        with[0] = '\0';
        with_globbed[0] = '\0';
        strcat(with, replacement_command);
        strcat(with, get_temp_dir());
        strcat(with, "/");
        strcat(with, get_uuid());
        strcpy(with_globbed, with);
        strcat(with_globbed, "*");
        strcat(with, extension);
        strcat(with_globbed, extension);
    }
    strcpy(tmp, orig);

    // [MODIF 2022]
    pyresult = PyList_New(0);
    PyObject *temp = Py_BuildValue("s", result);
    PyList_Append(pyresult, temp);
    PyList_Append(pyresult, pyglobs);
    Py_XDECREF(pyglobs);
    Py_XDECREF(temp);
    return pyresult;
}

PyObject *
gmic_py_display_with_matplotlib_or_ipython(PyObject *image_files_glob_strings)
{
    // fprintf(stdout, "gmic_py_display_with_matplotlib_or_ipython\n");
    if (!PyList_Check(image_files_glob_strings)) {
        PyErr_Format(GmicException, "input globs list is not a Python list");

        return NULL;
    }
    PyObject *ipython_display_module = NULL;
    PyObject *matplotlib_pyplot_module = NULL;
    PyObject *matplotlib_image_module = NULL;
    PyObject *glob_module = NULL;
    PyObject *image_glob_str = NULL;
    PyObject *image_expanded_filenames = NULL;
    PyObject *all_image_expanded_filenames = NULL;
    PyObject *image = NULL;
    PyObject *display_result = NULL;
    unsigned int nb_subplots = 0;
    int matplotlib_subplot_id = 1;
    bool use_matplotlib = false;
    bool use_ipython = false;
    unsigned int i = 0;
    unsigned int j = 0;
    matplotlib_pyplot_module = PyImport_ImportModule("matplotlib.pyplot");
    if (matplotlib_pyplot_module == NULL) {
        use_matplotlib = false;
        ipython_display_module = PyImport_ImportModule("IPython.core.display");
        if (ipython_display_module == NULL) {
            use_ipython = false;
            PyErr_Clear();
            PyErr_Format(GmicException,
                         "Could not use matplotlib neither ipython to try to "
                         "display images");
            return NULL;
        }
        else {
            PyErr_Clear();
            use_ipython = true;
        }
    }
    else {
        use_matplotlib = true;
        use_ipython = false;
        matplotlib_image_module = PyImport_ImportModule("matplotlib.image");
    }

    all_image_expanded_filenames = PyList_New(0);
    glob_module = PyImport_ImportModule("glob");
    for (i = 0; i < PyList_Size(image_files_glob_strings); i++) {
        // display(Image('image_path...', unconfined=True))
        image_glob_str = PyList_GetItem(image_files_glob_strings, i);
        image_expanded_filenames =
            PyObject_CallMethod(glob_module, "glob", "O", image_glob_str);
        for (j = 0; j < PyList_Size(image_expanded_filenames); j++) {
            PyList_Append(all_image_expanded_filenames,
                          PyList_GetItem(image_expanded_filenames, j));
        }
    }
    // Sort files by unique ID otherwise they will be in some kind of mess,
    // because glob.glob does not sort
    PyList_Sort(all_image_expanded_filenames);

    nb_subplots = PyList_Size(all_image_expanded_filenames);

    for (j = 0; j < nb_subplots; j++) {
        if (use_matplotlib) {
            image = PyObject_CallMethod(
                matplotlib_image_module, "imread", "O",
                PyList_GetItem(all_image_expanded_filenames, j));
            if (!image) {
                return image;
            }
            display_result =
                PyObject_CallMethod(matplotlib_pyplot_module, "subplot", "iii",
                                    nb_subplots, 1, matplotlib_subplot_id++);
            if (!display_result) {
                return display_result;
            }

            display_result = PyObject_CallMethod(matplotlib_pyplot_module,
                                                 "imshow", "O", image);
            if (!display_result) {
                return display_result;
            }
        }
        else if (use_ipython) {
            image = PyObject_CallMethod(
                ipython_display_module, "Image", "O",
                PyList_GetItem(all_image_expanded_filenames, j));
            if (image == NULL) {
                return image;
            }
            display_result = PyObject_CallMethod(ipython_display_module,
                                                 "display", "O", image);
            if (display_result == NULL) {
                return display_result;
            }
        }
        else {
            PyErr_Format(GmicException,
                         "Logic error: matplotlib or ipython should have "
                         "been imported at this point.");
            return NULL;
        }
    }
    // Matplolib requires only one single image display call, for multiple
    // images
    if (use_matplotlib) {
        display_result =
            PyObject_CallMethod(matplotlib_pyplot_module, "show", NULL);
        if (!display_result) {
            return display_result;
        }

        Py_XDECREF(all_image_expanded_filenames);
        Py_XDECREF(image_expanded_filenames);
        Py_XDECREF(image_glob_str);
        Py_XDECREF(image);
    }

    Py_XDECREF(ipython_display_module);
    Py_XDECREF(matplotlib_pyplot_module);
    Py_XDECREF(glob_module);

    return display_result;
}

PyObject *
autoload_wurlitzer_into_ipython()
{
    // fprintf(stdout, "autoload_wurlitzer_into_ipython\n");
    PyObject *wurlitzer_module = NULL;
    PyObject *ipython_module = NULL;
    PyObject *ipython_handler = NULL;
    PyObject *ipython_run_line_magic_result = NULL;
    PyObject *ipython_loaded_extensions = NULL;

    if (cimg_OS == 1) {  // UNIX OSes
        wurlitzer_module = PyImport_ImportModule("wurlitzer");
        if (wurlitzer_module == NULL) {
            PySys_WriteStdout(
                "gmic-py: If you do not see any text for G'MIC "
                "'print' or "
                "'display' commands, you could '!pip install "
                "wurlitzer' "
                "and if under an IPython environment, run the "
                "'%%load_ext "
                "wurlitzer' macro. See "
                "https://github.com/myselfhimself/gmic-py/issues/"
                "64\n");
            PyErr_Clear();
        }
        else {  // if wurlitzer module could be imported
            ipython_module = PyImport_ImportModule("IPython");
            if (ipython_module == NULL) {
                PyErr_Clear();
                Py_RETURN_NONE;
            }
            else {  // If IPython module found
                ipython_handler =
                    PyObject_CallMethod(ipython_module, "get_ipython", NULL);
                if (ipython_handler == NULL) {
                    PyErr_Clear();
                    return NULL;
                }
                // Skip any wurlitzer imported if not in an IPython context, or
                // if an IPython terminal
                else if (ipython_handler == Py_None ||
                         !PyObject_HasAttrString(ipython_handler, "kernel")) {
                    // See
                    // https://github.com/myselfhimself/gmic-py/issues/63#issuecomment-703533397
                    Py_XDECREF(ipython_handler);
                    Py_XDECREF(wurlitzer_module);
                    Py_XDECREF(ipython_module);
                    Py_RETURN_NONE;
                }
                else {
                    ipython_loaded_extensions = PyObject_GetAttrString(
                        PyObject_GetAttrString(ipython_handler,
                                               "extension_manager"),
                        "loaded");
                    if (ipython_loaded_extensions == NULL) {
                        PyErr_Clear();
                    }
                    else {
                        // if wurlitzer extension not loaded yet into
                        // IPython, try to load it
                        if (PySet_Contains(
                                ipython_loaded_extensions,
                                PyUnicode_FromString("wurlitzer")) == 0) {
                            ipython_run_line_magic_result =
                                PyObject_CallMethod(ipython_handler,
                                                    "run_line_magic", "ss",
                                                    "load_ext", "wurlitzer");
                            if (ipython_run_line_magic_result == NULL) {
                                PySys_WriteStdout(
                                    "gmic-py: managed to find IPython "
                                    "but "
                                    "could not call the '%%load_ext "
                                    "wurltizer "
                                    "macro for you. If you '!pip "
                                    "install "
                                    "wurlitzer' or install "
                                    "'wurlitzer' in "
                                    "your virtual environment, "
                                    "gmic-py will "
                                    "try to load it for you "
                                    "automatically.\n");
                                PyErr_Clear();
                            }
                            else {
                                PySys_WriteStderr(
                                    "gmic-py: wurlitzer found (for "
                                    "G'MIC "
                                    "stdout/stderr redirection) and "
                                    "enabled "
                                    "automatically through IPython "
                                    "'%%load_ext wurlitzer'.\n");
                            }
                        }
                    }
                }
            }
        }
    }
    else {  // Non-UNIX OSes
        PySys_WriteStdout(
            "You are not on a UNIX-like OS and unless you do "
            "have a "
            "side-window console, you shall not see any text "
            "for "
            "G'MIC 'print' or 'display' commands output. Hope "
            "you can "
            "accept it so. See "
            "https://github.com/myselfhimself/gmic-py/issues/"
            "64\n");
    }

    Py_XDECREF(wurlitzer_module);
    Py_XDECREF(ipython_module);
    Py_XDECREF(ipython_handler);
    Py_XDECREF(ipython_run_line_magic_result);
    return ipython_run_line_magic_result;
}

// end gmic_py_jupyter_ipython_display
#endif

static PyObject *
run_impl(PyObject *self, PyObject *args, PyObject *kwargs)
{
    char const *keywords[] = {"command", "images", "image_names", "nodisplay",
                              nullptr};
    PyObject *input_gmic_images = nullptr;
    PyObject *input_gmic_image_names = nullptr;
    char *commands_line = nullptr;
#ifdef gmic_py_jupyter_ipython_display
    static bool no_display_checked = false;
    static bool no_display_available = false;
    PyObject *commands_line_display_to_ouput_result = NULL;
    PyObject *ipython_matplotlib_display_result = NULL;
#endif
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "s|OO", const_cast<char **>(keywords),
            &commands_line, &input_gmic_images, &input_gmic_image_names)) {
        return nullptr;
    }

    try {
        PyObject *iter = nullptr;
        gmic_list<char> image_names;
        Py_XINCREF(input_gmic_images);
        Py_XINCREF(input_gmic_image_names);

#ifdef gmic_py_jupyter_ipython_display
        // Use a special way of displaying images only if the OS's display
        // is not available
        if (!no_display_checked) {
            no_display_checked = true;
            no_display_available = (getenv("DISPLAY") == NULL);
            if (no_display_available) {
                PySys_WriteStdout("gmic-py: Working in display-less mode.\n");
            }
        }

        if (no_display_available) {
            // Provide a fallback for gmic "display" command (without
            // supporting arguments) The idea is to replace all
            // occurences of "display" by "display output someprefix.png"
            commands_line_display_to_ouput_result =
                gmic_py_str_replace_display_to_output(commands_line,
                                                      (char *)".png");
            if (commands_line_display_to_ouput_result == NULL) {
                Py_XDECREF(input_gmic_images);
                Py_XDECREF(input_gmic_image_names);
                // Pass exception upwards
                return NULL;
            }
            commands_line = (char *)PyUnicode_AsUTF8(
                PyList_GetItem(commands_line_display_to_ouput_result, 0));
        }
#endif

        // Grab image names or single image name and check typings
        if (input_gmic_image_names != nullptr) {
            char *current_image_name_raw;
            // If list of image names provided
            if (PyList_Check(input_gmic_image_names)) {
                PyObject *current_image_name;
                PyObject *name_iter = PyObject_GetIter(input_gmic_image_names);
                const int image_names_count = Py_SIZE(input_gmic_image_names);
                image_names.assign(image_names_count);
                int image_name_position = 0;

                while ((current_image_name = PyIter_Next(name_iter))) {
                    if (!PyUnicode_Check(current_image_name)) {
                        PyErr_Format(PyExc_TypeError,
                                     "'%.50s' input element found at position "
                                     "%d in "
                                     "'image_names' list is not a '%.400s'",
                                     Py_TYPE(current_image_name)->tp_name,
                                     image_name_position,
                                     PyUnicode_Type.tp_name);
                        Py_XDECREF(input_gmic_images);
                        Py_XDECREF(input_gmic_image_names);

                        return nullptr;
                    }

                    current_image_name_raw = const_cast<char *>(
                        PyUnicode_AsUTF8(current_image_name));
                    image_names[image_name_position].assign(
                        strlen(current_image_name_raw) + 1);
                    memcpy(image_names[image_name_position]._data,
                           current_image_name_raw,
                           image_names[image_name_position]._width);
                    image_name_position++;
                }

                // If single image name provided
            }
            else if (PyUnicode_Check(input_gmic_image_names)) {
                // Enforce also non-null single-GmicImage 'images'
                // parameter
                if (input_gmic_images != nullptr &&
                    Py_TYPE(input_gmic_images) != &PyGmicImageType) {
                    PyErr_Format(PyExc_TypeError,
                                 "'%.50s' 'images' parameter must be a "
                                 "'%.400s' if the "
                                 "'image_names' parameter is a bare '%.400s'.",
                                 Py_TYPE(input_gmic_images)->tp_name,
                                 PyGmicImageType.tp_name,
                                 PyUnicode_Type.tp_name);
                    Py_XDECREF(input_gmic_images);
                    Py_XDECREF(input_gmic_image_names);

                    return nullptr;
                }

                image_names.assign(1);
                current_image_name_raw = const_cast<char *>(
                    PyUnicode_AsUTF8(input_gmic_image_names));
                image_names[0].assign(strlen(current_image_name_raw) + 1);
                memcpy(image_names[0]._data, current_image_name_raw,
                       image_names[0]._width);
                // If neither a list of strings nor a single string
                // were provided, raise exception
            }
            else {
                PyErr_Format(PyExc_TypeError,
                             "'%.50s' 'image_names' parameter must be a list "
                             "of '%.400s'(s)",
                             Py_TYPE(input_gmic_image_names)->tp_name,
                             PyUnicode_Type.tp_name);

                // [MODIF 2022]
                Py_XDECREF(input_gmic_images);
                Py_XDECREF(input_gmic_image_names);
                Py_XDECREF(iter);

                return nullptr;
            }
        }

        if (input_gmic_images != nullptr) {
            gmic_list<T> images;
            // A/ If a list of images was provided
            if (PyList_Check(input_gmic_images)) {
                PyObject *current_image;
                int image_position = 0;
                images.assign(Py_SIZE(input_gmic_images));

                // Grab images into a proper gmic_list after checking
                // their typing
                iter = PyObject_GetIter(input_gmic_images);
                while ((current_image = PyIter_Next(iter))) {
                    // If gmic_list item type is not a GmicImage
                    if (Py_TYPE(current_image) != &PyGmicImageType) {
                        PyErr_Format(PyExc_TypeError,
                                     "'%.50s' input object found at "
                                     "position %d in "
                                     "'images' list is not a '%.400s'",
                                     Py_TYPE(current_image)->tp_name,
                                     image_position, PyGmicImageType.tp_name);

                        Py_XDECREF(input_gmic_images);
                        Py_XDECREF(input_gmic_image_names);

                        return nullptr;
                    }
                    // Fill our just created gmic_list at same index
                    // with gmic_image coming from Python
                    swap_gmic_image_into_gmic_list(
                        reinterpret_cast<PyGmicImage *>(current_image), images,
                        image_position);

                    // [MODIF 2022]
                    Py_XDECREF(current_image);
                    image_position++;
                }

                // Process images and names
                reinterpret_cast<PyGmic *>(self)->_gmic->run(
                    commands_line, images, image_names);

                // [MODIF 2022]
                // Prevent images auto-deallocation by G'MIC
                // image_position = 0;

                // Bring new images set back into the Python world
                // (change List items in-place) First empty the input
                // Python images List object from its items without
                // deleting it (empty list, same reference)
                PySequence_DelSlice(input_gmic_images, 0,
                                    PySequence_Length(input_gmic_images));

                cimglist_for(images, l)
                {
                    // On the fly python GmicImage build per
                    // https://stackoverflow.com/questions/4163018/create-an-object-using-pythons-c-api/4163055#comment85217110_4163055
                    PyObject *_data = PyBytes_FromStringAndSize(
                        reinterpret_cast<const char *>(images[l]._data),
                        static_cast<Py_ssize_t>(sizeof(T) *
                                                get_image_size(images[l])));

                    PyObject *new_gmic_image = PyObject_CallFunction(
                        reinterpret_cast<PyObject *>(&PyGmicImageType),
                        // The last argument is a p(redicate), ie.
                        // boolean..
                        // but Py_BuildValue() used by
                        // PyObject_CallFunction has a slightly
                        // different parameters format specification
                        "SIIIIi", _data, images[l]._width, images[l]._height,
                        images[l]._depth, images[l]._spectrum,
                        static_cast<int>(images[l]._is_shared));
                    if (new_gmic_image == nullptr) {
                        PyErr_Format(
                            PyExc_RuntimeError,
                            "Could not initialize GmicImage for "
                            "appending "
                            "it to provided 'images' parameter list.");
                        return nullptr;
                    }
                    // [MODIF 2022]
                    PyObject *temp = new_gmic_image;
                    PyList_Append(input_gmic_images, temp);
                    Py_XDECREF(temp);
                    Py_XDECREF(_data);
                }
                // [MODIF 2022]
                Py_XDECREF(iter);
                // B/ Else if a single GmicImage was provided
            }
            else if (Py_TYPE(input_gmic_images) == &PyGmicImageType) {
                images.assign(1);
                swap_gmic_image_into_gmic_list(
                    reinterpret_cast<PyGmicImage *>(input_gmic_images), images,
                    0);

                // Pipe the commands, our single image, and no image
                // names
                reinterpret_cast<PyGmic *>(self)->_gmic->run(
                    commands_line, images, image_names);

                // Alter the original image only if the gmic_image list
                // has not been downsized to 0 elements this may happen
                // with eg. a rm[0] G'MIC command We must prevent this,
                // because a 'core dumped' happens otherwise
                if (images._width > 0) {
                    swap_gmic_list_item_into_gmic_image(
                        images, 0,
                        reinterpret_cast<PyGmicImage *>(input_gmic_images));
                }
                else {
                    PyErr_Format(PyExc_RuntimeError,
                                 "'%.50s' 'images' single-element parameter "
                                 "was removed by your G\'MIC command. It was "
                                 "probably emptied, your optional "
                                 "'image_names' list is untouched.",
                                 Py_TYPE(input_gmic_images)->tp_name,
                                 PyGmicImageType.tp_name,
                                 PyGmicImageType.tp_name);
                    Py_XDECREF(input_gmic_images);
                    Py_XDECREF(input_gmic_image_names);

                    return nullptr;
                }
            }
            // Else if provided 'images' type is unknown, raise Error
            else {
                PyErr_Format(PyExc_TypeError,
                             "'%.50s' 'images' parameter must be a "
                             "'%.400s', or list "
                             "of either '%.400s'(s)",
                             Py_TYPE(input_gmic_images)->tp_name,
                             PyGmicImageType.tp_name, PyGmicImageType.tp_name);
                Py_XDECREF(input_gmic_images);
                Py_XDECREF(input_gmic_image_names);

                return nullptr;
            }

            // If a correctly-typed image names parameter was provided,
            // even if wrongly typed, let us update its Python object
            // in place, to mirror any kind of changes that may have
            // taken place in the gmic_list of image names
            if (input_gmic_image_names != nullptr) {
                // i) If a list parameter was provided
                if (PyList_Check(input_gmic_image_names)) {
                    // First empty the input Python image names list
                    PySequence_DelSlice(
                        input_gmic_image_names, 0,
                        PySequence_Length(input_gmic_image_names));
                    // Add image names from the Gmic List of names
                    cimglist_for(image_names, l)
                    {
                        // [MODIF 2022]
                        PyObject *temp = PyUnicode_FromString(image_names[l]);
                        PyList_Append(input_gmic_image_names, temp);
                        Py_XDECREF(temp);
                    }
                }
                // ii) If a str parameter was provided
                // Because of Python's string immutability, we will not
                // change the input string's content here :) :/
            }
        }
        else {  // If no gmic_images given
            reinterpret_cast<PyGmic *>(self)->_gmic->run<T>(commands_line);
        }

        Py_XDECREF(input_gmic_images);
        Py_XDECREF(input_gmic_image_names);
    }
    catch (gmic_exception &e) {
        PyErr_SetString(GmicException, e.what());
        return nullptr;
    }
    catch (std::exception &e) {
        PyErr_SetString(GmicException, e.what());
        return nullptr;
    }
#ifdef gmic_py_jupyter_ipython_display
    // Use a special way of displaying only if the OS's display is not
    // available
    if (no_display_available) {
        // Provide a fallback for gmic "display" command (without
        // supporting arguments) The idea is to replace all occurences
        // of "display" by "output someprefix.png display in ipython
        ipython_matplotlib_display_result =
            gmic_py_display_with_matplotlib_or_ipython(
                PyList_GetItem(commands_line_display_to_ouput_result, 1));
        if (ipython_matplotlib_display_result == NULL) {
            // If we are not within a IPython environment, this is OK
            // Let us just print the exception without throwing it further
            // This case typically happens in readthedocs.org for gmic-sphinx
            PyErr_Print();
        }
    }
    Py_XDECREF(commands_line_display_to_ouput_result);
    Py_XDECREF(ipython_matplotlib_display_result);
#endif

    Py_RETURN_NONE;
}

#ifdef gmic_py_numpy
/**
 * Predictable Python 3.x 'numpy' module importer.
 */
PyObject *
import_numpy_module()
{
    PyObject *numpy_module = PyImport_ImportModule("numpy");

    // exit raising numpy_module import exception
    if (!numpy_module) {
        PyErr_Clear();
        // [MODIF 2022]
        Py_XDECREF(numpy_module);
        return PyErr_Format(GmicException,
                            "The 'numpy' module cannot be imported. Is it "
                            "installed or in your Python path?");
    }

    return numpy_module;
}

/*
 * GmicImage class method from_numpy_helper().
 * This factory class method generates a G'MIC Image from a
 * numpy.ndarray.
 *
 *  GmicImage.from_numpy_helper(obj: numpy.ndarray, deinterleave=True,
 * permute="xyzc": bool) -> GmicImage
 */
static PyObject *
PyGmicImage_from_numpy_helper(PyObject *, PyObject *args, PyObject *kwargs)
{
    PyObject *py_arg_deinterleave = nullptr;
    const auto py_arg_deinterleave_default =
        Py_True;  // Will deinterleave the incoming numpy.ndarray by default
    PyObject *py_arg_ndarray = nullptr;
    char const *keywords[] = {"numpy_array", "deinterleave", "permute",
                              nullptr};
    char *arg_permute = nullptr;
    PyObject *numpy_module = import_numpy_module();
    if (!numpy_module)
        return nullptr;

    PyObject *ndarray_type = PyObject_GetAttrString(numpy_module, "ndarray");

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O!|O!s", const_cast<char **>(keywords),
            reinterpret_cast<PyTypeObject *>(ndarray_type), &py_arg_ndarray,
            &PyBool_Type, &py_arg_deinterleave, &arg_permute))
        return nullptr;

    py_arg_deinterleave = py_arg_deinterleave == nullptr
                              ? py_arg_deinterleave_default
                              : py_arg_deinterleave;

    Py_XINCREF(py_arg_ndarray);
    Py_XINCREF(py_arg_deinterleave);

    // Get number of dimensions and ensure we are >=1D <=4D
    const auto ndarray_ndim =
        PyLong_AsSize_t(PyObject_GetAttrString(py_arg_ndarray, "ndim"));
    if (ndarray_ndim < 1 || ndarray_ndim > 4) {
        PyErr_Format(GmicException,
                     "Provided 'data' of type 'numpy.ndarray' must be between "
                     "1D and 4D ('data.ndim'=%d).",
                     ndarray_ndim);
        // [MODIF 2022]
        Py_XDECREF(py_arg_ndarray);
        Py_XDECREF(py_arg_deinterleave);
        return nullptr;
    }

    // Get input ndarray.dtype and prevent non-integer/float/bool data
    // types to be processed
    PyObject *ndarray_dtype = PyObject_GetAttrString(py_arg_ndarray, "dtype");
    // Ensure dtype kind is a number we can convert (from dtype values
    // here:
    // https://numpy.org/doc/1.18/reference/generated/numpy.dtype.kind.html#numpy.dtype.kind)
    PyObject *ndarray_dtype_kind =
        PyObject_GetAttrString(ndarray_dtype, "kind");
    if (strchr("biuf", static_cast<int>(PyUnicode_ReadChar(ndarray_dtype_kind,
                                                           0))) == nullptr) {
        PyErr_Format(PyExc_TypeError,
                     "Parameter 'data' of type 'numpy.ndarray' does not "
                     "contain numbers ie. its 'dtype.kind'(=%U) is not one of "
                     "'b', 'i', 'u', 'f'.",
                     ndarray_dtype_kind);
        // [MODIF 2022]
        Py_XDECREF(ndarray_dtype);
        Py_XDECREF(ndarray_dtype_kind);
        Py_XDECREF(py_arg_ndarray);
        Py_XDECREF(py_arg_deinterleave);
        return nullptr;
    }

    // Using an 'ndarray.astype' array casting operation first into
    // G'MIC's core point type T <=> float32 With a
    // memory-efficient-'ndarray.view' instead of copy-obligatory
    // 'ndarray.astype' conversion, we might get the following error:
    // ValueError: When changing to a larger dtype, its size must be a
    // divisor of the total size in bytes of the last axis of the
    // array. So, using 'astype' is the most stable, less
    // memory-efficient way
    // :-) :-/

    // [MODIF 2022]
    PyObject *attr_str_for_float =
        PyObject_GetAttrString(numpy_module, "float32");
    PyObject *float32_ndarray =
        PyObject_CallMethod(py_arg_ndarray, "astype", "O", attr_str_for_float);
    Py_XDECREF(attr_str_for_float);

    // Get unsqueezed shape of numpy array -> GmicImage width, height,
    // depth, spectrum Getting a shape with the most axes from array:
    // https://docs.scipy.org/doc/numpy-1.17.0/reference/generated/numpy.atleast_3d.html#numpy.atleast_3d
    // Adding a depth axis using numpy.expand_dims:
    // https://docs.scipy.org/doc/numpy-1.17.0/reference/generated/numpy.expand_dims.html
    // (numpy tends to squeeze dimensions when calling the standard
    // array().shape, we circumvent this)
    PyObject *ndarray_as_3d_unsqueezed_view =
        PyObject_CallMethod(numpy_module, "atleast_3d", "O", float32_ndarray);
    PyObject *ndarray_as_3d_unsqueezed_view_expanded_dims =
        PyObject_CallMethod(numpy_module, "expand_dims", "OI",
                            ndarray_as_3d_unsqueezed_view,
                            2);  // Adding z axis if absent
    // After this the shape should be (w, h, 1, 3)
    PyObject *ndarray_shape_tuple = PyObject_GetAttrString(
        ndarray_as_3d_unsqueezed_view_expanded_dims, "shape");

    const auto _height =
        PyLong_AsSize_t(PyTuple_GetItem(ndarray_shape_tuple, 0));
    const auto _width =
        PyLong_AsSize_t(PyTuple_GetItem(ndarray_shape_tuple, 1));
    const auto _depth =
        PyLong_AsSize_t(PyTuple_GetItem(ndarray_shape_tuple, 2));
    const auto _spectrum =
        PyLong_AsSize_t(PyTuple_GetItem(ndarray_shape_tuple, 3));

    const auto py_gmicimage_to_fill =
        reinterpret_cast<PyGmicImage *>(PyObject_CallFunction(
            reinterpret_cast<PyObject *>(&PyGmicImageType), "OIIII",
            Py_None,  // This empty _data buffer will be regenerated by the
                      // GmicImage constructor as a zero-filled bytes
                      // object
            _width, _height, _depth, _spectrum));

    PyObject *ndarray_data_bytesObj =
        PyObject_CallMethod(ndarray_as_3d_unsqueezed_view, "tobytes", nullptr);
    auto ndarray_data_bytesObj_ptr =
        reinterpret_cast<T *>(PyBytes_AsString(ndarray_data_bytesObj));

    // no deinterleaving
    if (!PyObject_IsTrue(py_arg_deinterleave)) {
        for (unsigned int c = 0; c < _spectrum; c++) {
            for (unsigned int z = 0; z < _depth; z++) {
                for (unsigned int y = 0; y < _height; y++) {
                    for (unsigned int x = 0; x < _width; x++) {
                        (*py_gmicimage_to_fill->_gmic_image)(x, y, z, c) =
                            *ndarray_data_bytesObj_ptr++;
                    }
                }
            }
        }
    }
    else {  // deinterleaving
        for (unsigned int z = 0; z < _depth; z++) {
            for (unsigned int y = 0; y < _height; y++) {
                for (unsigned int x = 0; x < _width; x++) {
                    for (unsigned int c = 0; c < _spectrum; c++) {
                        (*(py_gmicimage_to_fill->_gmic_image))(x, y, z, c) =
                            *(ndarray_data_bytesObj_ptr++);
                    }
                }
            }
        }
    }

    Py_XDECREF(py_arg_ndarray);
    Py_XDECREF(py_arg_deinterleave);
    Py_XDECREF(ndarray_dtype);
    Py_XDECREF(ndarray_dtype_kind);
    Py_XDECREF(float32_ndarray);
    Py_XDECREF(ndarray_as_3d_unsqueezed_view);
    Py_XDECREF(ndarray_as_3d_unsqueezed_view_expanded_dims);
    Py_XDECREF(ndarray_shape_tuple);
    Py_XDECREF(ndarray_data_bytesObj);
    Py_XDECREF(ndarray_type);
    Py_XDECREF(numpy_module);

    return reinterpret_cast<PyObject *>(py_gmicimage_to_fill);
}

static PyObject *
PyGmicImage_from_numpy(PyObject *cls, PyObject *args, PyObject *kwargs)
{
    char const *keywords[] = {"numpy_array", nullptr};
    PyObject *arg_np_array = nullptr;  // No defaults

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O", const_cast<char **>(keywords), &arg_np_array)) {
        return nullptr;
    }
    PyObject *a = PyTuple_New(0);
    PyObject *kw = PyDict_New();
    PyDict_SetItemString(kw, "numpy_array", arg_np_array);
    PyDict_SetItemString(kw, "deinterleave", Py_True);
    Py_XDECREF(arg_np_array);

    // [MODIF 2022]
    Py_XDECREF(args);
    PyObject *get_attr = PyObject_GetAttrString(cls, "from_numpy_helper");
    PyObject *ret = PyObject_Call(get_attr, a, kw);
    Py_XDECREF(get_attr);
    return ret;
}

static PyObject *
PyGmicImage_to_numpy(PyObject *self, PyObject *, PyObject *)
{
    PyObject *a = PyTuple_New(0);
    PyObject *kw = PyDict_New();
    PyDict_SetItemString(kw, "interleave", Py_True);

    // [MODIF 2022]
    PyObject *numpy_get_attr = PyObject_GetAttrString(self, "to_numpy_helper");
    PyObject *call_to_numpy = PyObject_Call(numpy_get_attr, a, kw);

    Py_XDECREF(a);
    Py_XDECREF(kw);
    Py_XDECREF(numpy_get_attr);

    return call_to_numpy;
}

static PyObject *
PyGmicImage_from_skimage(PyObject *cls, PyObject *args, PyObject *kwargs)
{
    char const *keywords[] = {"scikit_image", nullptr};
    PyObject *arg_scikit_image = nullptr;  // No defaults

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O",
                                     const_cast<char **>(keywords),
                                     &arg_scikit_image)) {
        return nullptr;
    }
    PyObject *a = PyTuple_New(0);
    PyObject *kw = PyDict_New();
    PyDict_SetItemString(kw, "numpy_array", arg_scikit_image);
    PyDict_SetItemString(kw, "deinterleave", Py_True);
    PyObject *py_permute_str = PyUnicode_FromString("zyxc");
    PyDict_SetItemString(kw, "permute", py_permute_str);

    return PyObject_Call(PyObject_GetAttrString(cls, "from_numpy_helper"), a,
                         kw);
}

static PyObject *
PyGmicImage_to_skimage(PyObject *self, PyObject *, PyObject *)
{
    PyObject *a = PyTuple_New(0);
    PyObject *kw = PyDict_New();
    PyDict_SetItemString(kw, "interleave", Py_True);
    PyObject *py_permute_str = PyUnicode_FromString("zyxc");
    PyDict_SetItemString(kw, "permute", py_permute_str);

    PyObject *numpy_get_attr = PyObject_GetAttrString(self, "to_numpy_helper");
    PyObject *call_to_numpy = PyObject_Call(numpy_get_attr, a, kw);

    Py_XDECREF(numpy_get_attr);
    Py_XDECREF(a);
    Py_XDECREF(kw);

    return call_to_numpy;
}

static PyObject *
PyGmicImage_from_PIL(PyObject *cls, PyObject *args, PyObject *kwargs)
{
    PyObject *numpy_mod;
    char const *keywords[] = {"pil_image", nullptr};
    PyObject *PIL_Image_mod;
    PyObject *arg_PIL_image = nullptr;  // No defaults

    if (!((numpy_mod = import_numpy_module()))) {
        return nullptr;
    }

    if (!((PIL_Image_mod = PyImport_ImportModule("PIL.Image")))) {
        return nullptr;
    }

    PyObject *PIL_Image_Image_class =
        PyObject_GetAttrString(PIL_Image_mod, "Image");

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O!", const_cast<char **>(keywords),
            reinterpret_cast<PyTypeObject *>(PIL_Image_Image_class),
            &arg_PIL_image)) {
        return nullptr;
    }
    PyObject *numpy_intermediate_array = PyObject_CallFunction(
        PyObject_GetAttrString(numpy_mod, "array"), "O", arg_PIL_image);
    if (!numpy_intermediate_array) {
        return nullptr;
    }
    PyObject *a = PyTuple_New(0);
    PyObject *kw = PyDict_New();
    PyDict_SetItemString(kw, "numpy_array", numpy_intermediate_array);
    PyDict_SetItemString(kw, "deinterleave", Py_True);
    PyObject *py_permute_str = PyUnicode_FromString("zyxc");
    PyDict_SetItemString(kw, "permute", py_permute_str);

    Py_DECREF(PIL_Image_Image_class);
    Py_DECREF(numpy_intermediate_array);
    Py_DECREF(PIL_Image_mod);
    Py_DECREF(numpy_mod);

    // [MODIF 2022]
    PyObject *numpy_get_attr =
        PyObject_GetAttrString(cls, "from_numpy_helper");
    PyObject *call_to_numpy = PyObject_Call(numpy_get_attr, a, kw);

    Py_XDECREF(numpy_get_attr);
    Py_XDECREF(a);
    Py_XDECREF(kw);

    return call_to_numpy;
}

static PyObject *
PyGmicImage_to_PIL(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *gmic_mod;
    PyObject *numpy_mod;
    PyObject *PIL_Image_mod;
    char const *keywords[] = {"astype", "squeeze_shape", "mode", nullptr};
    PyObject *arg_astype = nullptr;      // defaults to numpy.uint8
    unsigned int arg_squeeze_shape = 1;  // Defaults to true
    PyObject *arg_mode = nullptr;        // Defaults to 'RGB'

    if (!((gmic_mod = PyImport_ImportModule("gmic")))) {
        return nullptr;
    }

    if (!((numpy_mod = import_numpy_module()))) {
        return nullptr;
    }

    if (!((PIL_Image_mod = PyImport_ImportModule("PIL.Image")))) {
        return nullptr;
    }

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "|O!pO", const_cast<char **>(keywords), &PyType_Type,
            &arg_astype, &arg_squeeze_shape, &arg_mode)) {
        return nullptr;
    }

    if (arg_astype == nullptr) {
        arg_astype = PyObject_GetAttrString(numpy_mod, "uint8");
    }

    if (arg_mode == nullptr) {
        arg_mode = PyUnicode_FromString("RGB");
    }

    PyObject *a = PyTuple_New(0);
    PyObject *kw = PyDict_New();
    PyDict_SetItemString(kw, "interleave", Py_True);
    PyDict_SetItemString(kw, "astype", arg_astype);
    if (arg_squeeze_shape) {
        PyDict_SetItemString(kw, "squeeze_shape", Py_True);
    }
    PyObject *py_permute_str = PyUnicode_FromString("zyxc");
    PyDict_SetItemString(kw, "permute", py_permute_str);

    PyObject *test = PyObject_GetAttrString(self, "to_numpy_helper");

    PyObject *prePIL_np_array = PyObject_Call(test, a, kw);

    Py_XDECREF(test);
    if (!prePIL_np_array) {
        return nullptr;
    }

    Py_DECREF(gmic_mod);
    Py_DECREF(numpy_mod);
    Py_DECREF(PIL_Image_mod);
    Py_DECREF(py_permute_str);
    Py_DECREF(kw);
    Py_DECREF(a);
    Py_XDECREF(arg_astype);

    // [MODIF 2022]
    PyObject *part = PyObject_GetAttrString(PIL_Image_mod, "fromarray");
    PyObject *funct =
        PyObject_CallFunction(part, "OO", prePIL_np_array, arg_mode);

    Py_XDECREF(part);
    Py_XDECREF(prePIL_np_array);

    return funct;
}

// End of ifdef gmic_py_numpy
#endif

/** Instancing of any c++ gmic::gmic G'MIC language interpreter object
 * (Python: gmic.Gmic) **/
PyObject *
PyGmic_new(PyTypeObject *subtype, PyObject *args, PyObject *kwargs)
{
    auto *self = reinterpret_cast<PyGmic *>(subtype->tp_alloc(subtype, 0));

    // Init resources folder.
    if (!gmic::init_rc()) {
        PyErr_Format(GmicException,
                     "Unable to create G'MIC resources folder.");
    }

    // Load general and user scripts if they exist
    // Since this project is a library the G'MIC "update" command that
    // runs an internet download, is never triggered the user should
    // run it him/herself.
    self->_gmic->run<T>("m $_path_rc/update$_version.gmic");
    self->_gmic->run<T>("m $_path_user");

    // If parameters are provided, pipe them to our run() method, and
    // do only exceptions raising without returning anything if things
    // go well
    if (args != Py_None &&
        ((args && static_cast<int>(PyTuple_Size(args)) > 0) ||
         (kwargs && static_cast<int>(PyDict_Size(kwargs)) > 0))) {
        if (run_impl(reinterpret_cast<PyObject *>(self), args, kwargs) ==
            nullptr) {
            return nullptr;
        }
    }

    return reinterpret_cast<PyObject *>(self);
}

PyDoc_STRVAR(constexpr run_impl_doc,
             "Gmic.run(command, images=None, image_names=None)\n\
Run G'MIC interpreter following a G'MIC language command(s) string, on 0 or more namable ``GmicImage`` items.\n\n\
Note (single-image short-hand calling): if ``images`` is a ``GmicImage``, then ``image_names`` must be either a ``str`` or be omitted.\n\n\
Example:\n\
    Here is a long example describing several use cases::\n\n\
        import gmic\n\
        import struct\n\
        import random\n\
        instance1 = gmic.Gmic('echo_stdout \\'instantiation and run all in one\\')\n\
        instance2 = gmic.Gmic()\n\
        instance2.run('echo_stdout \\'hello world\\'') # G'MIC command without images parameter\n\
        a = gmic.GmicImage(struct.pack(*('256f',) + tuple([random.random() for a in range(256)])), 16, 16) # Build 16x16 greyscale image\n\
        instance2.run('blur 12,0,1 resize 50%,50%', a) # Blur then resize the image\n\
        a._width == a._height == 8 # The image is half smaller\n\
        instance2.run('display', a) # If you have X11 enabled (linux only), show the image in a window\n\
        image_names = ['img_' + str(i) for i in range(10)] # You can also name your images if you have several (optional)\n\
        images = [gmic.GmicImage(struct.pack(*((str(w*h)+'f',) + (i*2.0,)*w*h)), w, h) for i in range(10)] # Prepare a list of image\n\
        instance1.run('add 1 print', images, image_names) # And pipe those into the interpreter\n\
        instance1.run('blur 10,0,1 print', images[0], 'my_pic_name') # Short-hand 1-image calling style\n\n\
Args:\n\
    command (str): An image-processing command in the G'MIC language\n\
    images (Optional[Union[List[gmic.GmicImage], gmic.GmicImage]]): A list of ``GmicImage`` items that G'MIC will edit in place, or a single ``gmic.GmicImage`` which will used for input only. Defaults to None.\n\
        Put a list variable here, not a plain ``[]``.\n\
        If you pass a list, it can be empty if you intend to fill or complement it using your G'MIC command.\n\
    image_names (Optional[List<str>]): A list of names for the images, defaults to None.\n\
        In-place editing by G'MIC can happen, you might want to pass your list as a variable instead.\n\
\n\
Returns:\n\
    None: Returns ``None`` or raises a ``GmicException``.\n\
\n\
Raises:\n\
    GmicException: This translates' G'MIC C++ same-named exception. Look at the exception message for details.");

static PyMethodDef PyGmic_methods[] = {
    {"run", reinterpret_cast<PyCFunction>(run_impl),
     METH_VARARGS | METH_KEYWORDS, run_impl_doc},
    {nullptr} /* Sentinel */
};

// ------------ G'MIC IMAGE BINDING ----//
PyObject *
PyGmicImage_new(PyTypeObject *subtype, PyObject *args, PyObject *kwargs)
{
    unsigned int _width =
        1;  // Number of image columns (dimension along the X-axis)
    unsigned int _height =
        1;  // Number of image lines (dimension along the Y-axis)
    unsigned int _depth =
        1;  // Number of image slices (dimension along the Z-axis)
    unsigned int _spectrum = 1;  // Number of image channels (dimension along
                                 // the C-axis) All integer parameters
                                 // multiplied together,
    // will help for allocating (ie. assign()ing)
    int _is_shared = 0;  // Whether image should be shared across gmic
                         // operations (if true,
    // operations like resize will fail)
    PyObject *bytesObj = nullptr;  // Incoming bytes buffer object pointer

    char const *keywords[] = {"data",     "width",  "height", "depth",
                              "spectrum", "shared", nullptr};

    // Parameters parsing and checking
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "|OIIIIp", const_cast<char **>(keywords), &bytesObj,
            &_width, &_height, &_depth, &_spectrum, &_is_shared))
        return nullptr;

    auto *self =
        reinterpret_cast<PyGmicImage *>(subtype->tp_alloc(subtype, 0));
    if (self == nullptr) {
        return nullptr;
    }

    // Default bytesObj value to None
    if (bytesObj == nullptr) {
        bytesObj = Py_None;
    }
    else {
        Py_INCREF(bytesObj);
    }

    if (bytesObj != Py_None) {
        const bool bytesObj_is_bytes =
            static_cast<bool>(PyBytes_Check(bytesObj));
        if (!bytesObj_is_bytes) {
            PyErr_Format(PyExc_TypeError,
                         "Parameter 'data' must be a "
                         "pure-python 'bytes' buffer object.");
            // TODO pytest this
            Py_XDECREF(bytesObj);
            return nullptr;
        }
    }
    else {  // if bytesObj is None, attempt to set it as an empty bytes
            // object
        // following image dimensions
        // If dimensions are OK, create a pixels-count-zero-filled
        // bytesarray-based bytes object to be ingested by the _data
        // parameter
        if (_width >= 1 && _height >= 1 && _depth >= 1 && _spectrum >= 1) {
            auto *pybytes_type = reinterpret_cast<PyObject *>(&PyBytes_Type);
            PyObject *pybytearray_type = PyObject_CallFunction(
                reinterpret_cast<PyObject *>(&PyByteArray_Type), "I",
                _width * _height * _depth * _spectrum * sizeof(T), NULL);

            bytesObj = PyObject_CallFunction(pybytes_type, "O",
                                             pybytearray_type, NULL);

            // Py_INCREF(bytesObj);

            Py_XDECREF(pybytearray_type);
            // TODO pytest this
        }
        else {  // If dimensions are not OK, raise exception
            PyErr_Format(PyExc_TypeError,
                         "If you do not provide a 'data' parameter, make at "
                         "least all dimensions >=1.");
            // TODO pytest this
            Py_XDECREF(bytesObj);
            return nullptr;
        }
    }

    // Bytes object spatial dimensions vs. bytes-length checking
    Py_ssize_t dimensions_product = _width * _height * _depth * _spectrum;
    const Py_ssize_t _data_bytes_size = PyBytes_Size(bytesObj);
    if (static_cast<Py_ssize_t>(dimensions_product * sizeof(T)) !=
        _data_bytes_size) {
        PyErr_Format(PyExc_ValueError,
                     "GmicImage dimensions-induced buffer bytes size "
                     "(%d*%dB=%d) cannot be strictly negative or "
                     "different than the _data buffer size in bytes (%d)",
                     dimensions_product, sizeof(T),
                     dimensions_product * sizeof(T), _data_bytes_size);
        Py_XDECREF(bytesObj);
        return nullptr;
    }

    // Importing input data to an internal buffer
    try {
        self->_gmic_image->assign(_width, _height, _depth, _spectrum);
        self->_gmic_image->_is_shared = _is_shared;
    }
    // Ugly exception catching, probably to catch a
    // cimg::GmicInstanceException()
    catch (...) {
        dimensions_product = _width * _height * _depth * _spectrum;
        PyErr_Format(PyExc_MemoryError,
                     "Allocation error in "
                     "GmicImage::assign(_width=%d,_height=%d,_depth=%d,_"
                     "spectrum=%d), "
                     "are you requesting too much memory (%d bytes)?",
                     _width, _height, _depth, _spectrum,
                     dimensions_product * sizeof(T));
        Py_XDECREF(bytesObj);
        return nullptr;
    }

    memcpy(self->_gmic_image->_data, PyBytes_AsString(bytesObj),
           PyBytes_Size(bytesObj));

    // [MODIF 2022]
    Py_XDECREF(bytesObj);
    Py_XDECREF(args);

    return reinterpret_cast<PyObject *>(self);
}

static PyObject *
PyGmicImage_repr(PyGmicImage *self)
{
    return PyUnicode_FromFormat(
        "<%s object at %p with _data address at %p, w=%d h=%d d=%d "
        "s=%d "
        "shared=%d>",
        Py_TYPE(self)->tp_name, self, self->_gmic_image->_data,
        self->_gmic_image->_width, self->_gmic_image->_height,
        self->_gmic_image->_depth, self->_gmic_image->_spectrum,
        self->_gmic_image->_is_shared);
}

static PyObject *
PyGmicImage_call(PyObject *self, PyObject *args, PyObject *kwargs)
{
    const char *keywords[] = {"x", "y", "z", "s", nullptr};
    int x, y, z, c;
    x = y = z = c = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|iiii",
                                     const_cast<char **>(keywords), &x, &y, &z,
                                     &c)) {
        return nullptr;
    }

    return PyFloat_FromDouble(
        (*reinterpret_cast<PyGmicImage *>(self)->_gmic_image)(x, y, z, c));
}

static PyObject *
PyGmicImage_alloc(PyTypeObject *type, Py_ssize_t)
{
    auto *obj = static_cast<PyObject *>(PyObject_Malloc(type->tp_basicsize));
    reinterpret_cast<PyGmicImage *>(obj)->_gmic_image = new gmic_image<T>();
    GMIC_PY_LOG("PyGmicImage_alloc\n");
    PyObject_Init(obj, type);
    return obj;
}

static PyObject *
PyGmic_alloc(PyTypeObject *type, Py_ssize_t)
{
    auto *obj = static_cast<PyObject *>(PyObject_Malloc(type->tp_basicsize));
    reinterpret_cast<PyGmic *>(obj)->_gmic = new gmic();
    GMIC_PY_LOG("PyGmic_alloc\n");
    PyObject_Init(obj, type);
    return obj;
}

static void
PyGmicImage_dealloc(PyGmicImage *self)
{
    delete self->_gmic_image;
    self->_gmic_image = nullptr;
    GMIC_PY_LOG("PyGmicImage_dealloc\n");
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

static void
PyGmic_dealloc(PyGmic *self)
{
    delete self->_gmic;
    self->_gmic = nullptr;
    GMIC_PY_LOG("PyGmic_dealloc\n");
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

static PyObject *
module_level_run_impl(PyObject *, PyObject *args, PyObject *kwargs)
{
    PyObject *temp_gmic_instance = PyObject_CallObject(
        reinterpret_cast<PyObject *>(&PyGmicType), nullptr);

    Py_INCREF(temp_gmic_instance);

    // Return None or a Python exception flag
    PyObject *run_impl_result = run_impl(temp_gmic_instance, args, kwargs);

    Py_XDECREF(temp_gmic_instance);

    return run_impl_result;
}

PyDoc_STRVAR(constexpr module_level_run_impl_doc,
             R"DOC(run(command, images=None, image_names=None)
Run the G'MIC interpreter with a G'MIC language command(s) string, on 0 or more nameable GmicImage(s). This is a short-hand for calling ``gmic.Gmic().run`` with the exact same parameters signature.

Note (single-image short-hand calling): if ``images`` is a ``GmicImage``, then ``image_names`` must be either a ``str`` or be omitted.

Note (interpreter warm-up): calling ``gmic.run`` multiple times is inefficient as it spawns then drops a new G'MIC interpreter instance for every call. For better performance, you can tie a ``gmic.Gmic`` G'MIC interpreter instance to a variable instead and call its ``run`` method multiple times. Look at ``gmic.Gmic.run`` for more information.

Example:
    Several ways to use the module-level ``gmic.run()`` function::
        import gmic
        import struct
        import random
        gmic.run("echo_stdout 'hello world'") # G'MIC command without images parameter
        a = gmic.GmicImage(struct.pack(*('256f',) + tuple([random.random() for a in range(256)])), 16, 16) # Build 16x16 greyscale image
        gmic.run('blur 12,0,1 resize 50%,50%', a) # Blur then resize the image
        a._width == a._height == 8 # The image is half smaller
        gmic.run('display', a) # If you have X11 enabled (linux only), show the image in a window
        image_names = ['img_' + str(i) for i in range(10)] # You can also name your images if you have several (optional)
        images = [gmic.GmicImage(struct.pack(*((str(w*h)+'f',) + (i*2.0,)*w*h)), w, h) for i in range(10)] # Prepare a list of image
        gmic.run('add 1 print', images, image_names) # And pipe those into the interpreter
        gmic.run('blur 10,0,1 print', images[0], 'my_pic_name') # Short-hand 1-image calling style

Args:
    command (str): An image-processing command in the G'MIC language
    images (Optional[Union[List[gmic.GmicImage], gmic.GmicImage]]): A list of ``GmicImage`` items that G'MIC will edit in place, or a single ``gmic.GmicImage`` which will used for input only. Defaults to None.
        Put a list variable here, not a plain ``[]``.
        If you pass a list, it can be empty if you intend to fill or complement it using your G'MIC command.
    image_names (Optional[List<str>]): A list of names for the images, defaults to None.
        In-place editing by G'MIC can happen, you might want to pass your list as a variable instead.

Returns:
    None: Returns ``None`` or raises a ``GmicException``.

Raises:
    GmicException: This translates' G'MIC C++ same-named exception. Look at the exception message for details.)DOC");

static PyMethodDef gmic_methods[] = {
    {"run", reinterpret_cast<PyCFunction>(module_level_run_impl),
     METH_VARARGS | METH_KEYWORDS, module_level_run_impl_doc},
    {nullptr, nullptr, 0, nullptr}};

PyDoc_STRVAR(constexpr gmic_module_doc,
             "G'MIC image processing library Python binary module.\n\n"
             "Use ``gmic.run`` or ``gmic.Gmic`` to run G'MIC commands inside "
             "the G'MIC C++ interpreter, manipulate ``gmic.GmicImage`` which "
             "has ``numpy``/``PIL`` input/output support, assemble lists of "
             "``gmic.GmicImage`` items inside read-writeable pure-Python "
             "`list` objects.");

PyModuleDef gmic_module = {PyModuleDef_HEAD_INIT, "gmic", gmic_module_doc, 0,
                           gmic_methods};

PyDoc_STRVAR(
    constexpr PyGmicImage_doc,
    R"DOC(GmicImage(data=None, width=1, height=1, depth=1, spectrum=1, shared=False)

Simplified mapping of the C++ ``gmic_image`` type. Stores a binary buffer of data, a height, width, depth, spectrum.

Example:

    Several ways to use a GmicImage simply::

        import gmic
        empty_1x1x1_black_image = gmic.GmicImage() # or gmic.GmicImage(None,1,1,1,1) for example
        import struct
        i = gmic.GmicImage(struct.pack('2f', 0.0, 1.5), 1, 1) # 2D 1x1 image
        gmic.run('add 1', i) # GmicImage injection into G'MIC's interpreter
        i # Using GmicImage's repr() string representation
        # Output: <gmic.GmicImage object at 0x7f09bfb504f8 with _data address at 0x22dd5b0, w=1 h=1 d=1 s=1 shared=0>
        i(0,0) == 1.0 # Using GmicImage(x,y,z) pixel reading operator after initialization
        gmic.run('resize 200%,200%', i) # Some G'MIC operations may reallocate the image buffer in place without risk
        i._width == i._height == 2 # Use the _width, _height, _depth, _spectrum, _data, _data_str, _is_shared read-only attributes

Args:
    data (Optional[bytes]): Raw data for the image (must be a sequence of 4-bytes floats blocks, with as many blocks as all the dimensions multiplied together).
    width (Optional[int]): Image width in pixels. Defaults to 1.
    height (Optional[int]): Image height in pixels. Defaults to 1.
    depth (Optional[int]): Image height in pixels. Defaults to 1.
    spectrum (Optional[int]): Number of channels per pixel. Defaults to 1.
    shared (Optional[bool]): C++ option: whether the buffer should be shareable between several GmicImages and operations. Defaults to False.

Note:
    **GmicImage(x=0, y=0, z=0, s=0)**

    This instance method allows you to read pixels in a ``GmicImage`` for given coordinates.

    You can read, but cannot write pixel values by passing some or all coordinates the following way::

        import gmic
        images = []
        gmic.run("sp apples", images)
        image = images[0]
        print(image(0,2,0,2)) # or image(y=2,z=2)
        print(image(0,0,0,0)) # or image()
        for x in range(image._width):
            for y in range(image._height):
                for z in range(image._depth):
                    for c in range(image._spectrum):
                        print(image(x,y,z,c)))DOC");

static PyObject *
PyGmicImage_get_width(const PyGmicImage *self, void *)
{
    return PyLong_FromSize_t(self->_gmic_image->_width);
}

static PyObject *
PyGmicImage_get_height(const PyGmicImage *self, void *)
{
    return PyLong_FromSize_t(self->_gmic_image->_height);
}

static PyObject *
PyGmicImage_get_depth(const PyGmicImage *self, void *)
{
    return PyLong_FromSize_t(self->_gmic_image->_depth);
}

static PyObject *
PyGmicImage_get_spectrum(const PyGmicImage *self, void *)
{
    return PyLong_FromSize_t(self->_gmic_image->_spectrum);
}

static PyObject *
PyGmicImage_get_is_shared(const PyGmicImage *self, void *)
{
    return PyBool_FromLong(self->_gmic_image->_is_shared);
}

static PyObject *
PyGmicImage_get_data(const PyGmicImage *self, void *)
{
    // Py_FinalizeEx();
    return PyBytes_FromStringAndSize(
        reinterpret_cast<char *>(self->_gmic_image->_data),
        static_cast<Py_ssize_t>(sizeof(T) *
                                get_image_size(self->_gmic_image)));
}

static PyObject *
PyGmicImage_get_data_str(const PyGmicImage *self, void *)
{
    const unsigned int image_size = get_image_size(self->_gmic_image);
    PyObject *unicode_json = PyUnicode_New(image_size, 65535);

    for (unsigned int a = 0; a < image_size; a++) {
        PyUnicode_WriteChar(unicode_json, a,
                            static_cast<Py_UCS4>(self->_gmic_image->_data[a]));
    }

    return unicode_json;
}

PyGetSetDef PyGmicImage_getsets[] = {
    {"_data", /* name */
     reinterpret_cast<getter>(PyGmicImage_get_data),
     nullptr,       // no setter
     "_data bytes", /* doc */
     nullptr /* closure */},
    {"_data_str", /* name */
     reinterpret_cast<getter>(PyGmicImage_get_data_str),
     nullptr,                   // no setter
     "_data bytes decoded str", /* doc */
     nullptr /* closure */},
    {"_width", reinterpret_cast<getter>(PyGmicImage_get_width), nullptr,
     "_width", nullptr},
    {"_height", reinterpret_cast<getter>(PyGmicImage_get_height), nullptr,
     "_height", nullptr},
    {"_depth", reinterpret_cast<getter>(PyGmicImage_get_depth), nullptr,
     "_depth", nullptr},
    {"_spectrum", reinterpret_cast<getter>(PyGmicImage_get_spectrum), nullptr,
     "_spectrum", nullptr},
    {"_is_shared", reinterpret_cast<getter>(PyGmicImage_get_is_shared),
     nullptr, "_is_shared", nullptr},
    {nullptr}};

#ifdef gmic_py_numpy

/*
 * GmicImage object method to_numpy_helper().
 *
 * GmicImage().to_numpy_helper(astype=numpy.float32: numpy.dtype,
 * interleave=False: bool, squeeze_shape=False: bool) -> numpy.ndarray
 *
 */
static PyObject *
PyGmicImage_to_numpy_helper(const PyGmicImage *self, PyObject *args,
                            PyObject *kwargs)
{
    char const *keywords[] = {"astype", "interleave", "permute",
                              "squeeze_shape", nullptr};
    PyObject *_tmp_return_ndarray;
    // Defaults to numpy.float32
    PyObject *arg_astype = nullptr;
    int arg_interleave = -1;
    constexpr int arg_interleave_default =
        0;  // Will not interleave the final matrix by default
    int arg_squeeze_shape = -1;
    constexpr int arg_squeeze_shape_default =
        0;  // Will not squeeze shape by default
    char *arg_permute = nullptr;
    char arg_permute_default[] = "xyzc";
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "|Opsp", const_cast<char **>(keywords), &arg_astype,
            &arg_interleave, &arg_permute, &arg_squeeze_shape)) {
        return nullptr;
    }

    // Set default values for any unset parameter
    arg_interleave =
        arg_interleave == -1 ? arg_interleave_default : arg_interleave;
    arg_permute = arg_permute == nullptr ? arg_permute_default : arg_permute;
    arg_squeeze_shape = arg_squeeze_shape == -1 ? arg_squeeze_shape_default
                                                : arg_squeeze_shape;
    // arg_astype will be set to numpy.float32 by default a bit later on

    PyObject *ndarray_shape_list = PyList_New(0);
    PyList_Append(ndarray_shape_list,
                  PyLong_FromLong(self->_gmic_image->_width));
    PyList_Append(ndarray_shape_list,
                  PyLong_FromLong(self->_gmic_image->_height));
    PyList_Append(ndarray_shape_list,
                  PyLong_FromLong(self->_gmic_image->_depth));
    PyList_Append(ndarray_shape_list,
                  PyLong_FromLong(self->_gmic_image->_spectrum));

    PyObject *ndarray_transpose_list = PyList_New(0);

    // Build the .(re)shape (w,h,d,s) and .transpose (axes permutation)
    // tuples
    if (arg_permute) {
        if (strlen(arg_permute) != 4) {
            PyErr_Format(
                GmicException,
                "'permute' parameter should be 4-characters long, %d found.",
                strlen(arg_permute));
            return nullptr;
        }
        for (size_t permute_axis = 0; permute_axis < strlen(arg_permute);
             permute_axis++) {
            switch (arg_permute[permute_axis]) {
                case 'x':
                    PyList_Append(ndarray_transpose_list, PyLong_FromLong(0L));
                    break;
                case 'y':
                    PyList_Append(ndarray_transpose_list, PyLong_FromLong(1L));
                    break;
                case 'z':
                    PyList_Append(ndarray_transpose_list, PyLong_FromLong(2L));
                    break;
                case 'c':
                    PyList_Append(ndarray_transpose_list, PyLong_FromLong(3L));
                    break;
                default:
                    PyErr_Format(GmicException,
                                 "'permute' parameter should be made up of "
                                 "x,y,z and c characters, '%s' found.",
                                 arg_permute);
                    return nullptr;
            }
        }
    }
    PyObject *ndarray_shape_tuple = PyList_AsTuple(ndarray_shape_list);

    PyObject *numpy_module = import_numpy_module();
    if (!numpy_module)
        return nullptr;

    PyObject *ndarray_type = PyObject_GetAttrString(numpy_module, "ndarray");

    PyObject *float32_dtype = PyObject_GetAttrString(numpy_module, "float32");
    const size_t buffer_size = sizeof(T) * get_image_size(self->_gmic_image);
    auto *numpy_buffer = static_cast<float *>(malloc(buffer_size));
    float *ndarray_bytes_buffer_ptr = numpy_buffer;
    // If interleaving is needed, copy the gmic_image buffer towards
    // numpy by interleaving RRR,GGG,BBB into RGB,RGB,RGB

    if (arg_interleave) {
        for (unsigned int x = 0; x < self->_gmic_image->_width; x++) {
            for (unsigned int y = 0; y < self->_gmic_image->_height; y++) {
                for (unsigned int z = 0; z < self->_gmic_image->_depth; z++) {
                    for (unsigned int c = 0; c < self->_gmic_image->_spectrum;
                         c++) {
                        (*ndarray_bytes_buffer_ptr++) =
                            (*(self->_gmic_image))(x, y, z, c);
                    }
                }
            }
        }
    }
    else {
        // If deinterleaving is not needed, since this is G'MIC's
        // internal image shape, keep pixel data order as is and copy
        // it simply
        memcpy(numpy_buffer, self->_gmic_image->_data,
               get_image_size(self->_gmic_image) * sizeof(T));
    }
    PyObject *numpy_bytes_buffer =
        PyBytes_FromStringAndSize(reinterpret_cast<const char *>(numpy_buffer),
                                  static_cast<Py_ssize_t>(buffer_size));
    free(numpy_buffer);
    // class numpy.ndarray(<our shape>, dtype=<float32>, buffer=<our
    // bytes>, offset=0, strides=None, order=None)
    PyObject *return_ndarray =
        PyObject_CallFunction(ndarray_type, "OOS", ndarray_shape_tuple,
                              float32_dtype, numpy_bytes_buffer);

    // arg_astype should be according to ndarray.astype's
    // documentation, a string, python type or numpy.dtype delegating
    // this type check to the astype() method
    if (return_ndarray != nullptr && arg_astype != nullptr) {
        _tmp_return_ndarray = return_ndarray;

        // to_numpy_helper(astype=None) will not result in
        // numpy.ndarray.astype(None)==numpy.float64 but float32 instead!
        if (arg_astype == Py_None) {
            arg_astype = float32_dtype;
        }

        return_ndarray =
            PyObject_CallMethod(return_ndarray, "astype", "O", arg_astype);
        if (!return_ndarray) {
            PyErr_Format(GmicException,
                         "'%.50s' failed to run numpy.ndarray.astype.",
                         Py_TYPE(ndarray_type)->tp_name);

            return nullptr;
        }
        // Get rid of the uncast ndarray
        Py_DECREF(_tmp_return_ndarray);
    }

    if (arg_permute != nullptr) {
        GMIC_PY_LOG("permutting within to_numpy_helper");
        // Store the untransposed array aside
        _tmp_return_ndarray = return_ndarray;

        return_ndarray = PyObject_CallMethod(return_ndarray, "transpose", "O",
                                             ndarray_transpose_list);

        if (!return_ndarray) {
            PyErr_Format(GmicException,
                         "'%.50s' failed to run numpy.ndarray.transpose "
                         "(permute).",
                         Py_TYPE(ndarray_type)->tp_name);

            return nullptr;
        }
        // Get rid of the untransposed ndarray
        Py_DECREF(_tmp_return_ndarray);
    }

    if (arg_squeeze_shape) {
        _tmp_return_ndarray = return_ndarray;
        return_ndarray =
            PyObject_CallMethod(numpy_module, "squeeze", "O", return_ndarray);
        if (!return_ndarray) {
            PyErr_Format(GmicException, "'%.50s' failed to run numpy.squeeze.",
                         Py_TYPE(ndarray_type)->tp_name);
        }
        else {
            // Get rid of the unsqueezed ndarray
            Py_DECREF(_tmp_return_ndarray);
        }
    }

    Py_XDECREF(ndarray_type);
    Py_XDECREF(ndarray_shape_list);
    Py_XDECREF(ndarray_shape_tuple);
    Py_XDECREF(ndarray_transpose_list);
    Py_XDECREF(float32_dtype);
    Py_XDECREF(numpy_bytes_buffer);
    Py_XDECREF(numpy_module);

    return return_ndarray;
}
#endif
// end ifdef gmic_py_numpy

#ifdef gmic_py_numpy
PyDoc_STRVAR(constexpr PyGmicImage_from_numpy_doc,
             R"DOC(GmicImage.from_numpy(numpy_array)

Make a GmicImage from a 1-4 dimensions numpy.ndarray. Simplified version of ``GmicImage.from_numpy_helper`` with ``deinterleave=True``.


Args:
    numpy_array (numpy.ndarray): A non-empty 1D-4D Numpy array.

Returns:
    GmicImage: A new ``GmicImage`` based the input ``numpy.ndarray`` data.

Raises:
    GmicException, TypeError: Look at the exception message for details. Matrices with dimensions <1D or >4D will be rejected.)DOC");

PyDoc_STRVAR(constexpr PyGmicImage_to_numpy_doc,
             R"DOC(GmicImage.to_numpy()

Make a numpy.ndarray from a GmicImage. Simplified version of ``GmicImage.to_numpy_helper`` with ``interleave=True``.

Returns:
    numpy.ndarray: A new ``numpy.ndarray`` based the input ``GmicImage`` data.)DOC");

PyDoc_STRVAR(
    constexpr PyGmicImage_from_numpy_helper_doc,
    R"DOC(GmicImage.from_numpy_helper(numpy_array, deinterleave=False, permute='')

Make a GmicImage from a 1-4 dimensions numpy.ndarray.

G'MIC works with (width, height, depth, spectrum/channels) matrix layout, with 32bit-float pixel values deinterleaved (ie. RRR,GGG,BBB).
If your matrix is less than 4D, G'MIC will tentatively add append void dimensions to it (eg. for a shape of (3,1) -> (3,1,1,1)). You can avoid this by using ``numpy.expand_dims`` or ``numpy.atleast_*d`` functions yourself first.
If your pixel values (ie. ``numpy.ndarray.dtype``) are not in a ``float32`` format, G'MIC will tentatively call ``numpy.astype(numpy_array, numpy.float32)`` to cast its contents first.

Example:

    Several ways to use a GmicImage simply::

        import gmic
        empty_1x1x1_black_image = gmic.GmicImage() # or gmic.GmicImage(None,1,1,1,1) for example
        import struct
        i = gmic.GmicImage(struct.pack('2f', 0.0, 1.5), 1, 1) # 2D 1x1 image
        gmic.run('add 1', i) # GmicImage injection into G'MIC's interpreter
        i # Using GmicImage's repr() string representation
        # Output: <gmic.GmicImage object at 0x7f09bfb504f8 with _data address at 0x22dd5b0, w=1 h=1 d=1 s=1 shared=0>
        i(0,0) == 1.0 # Using GmicImage(x,y,z) pixel reading operator after initialization
        gmic.run('resize 200%,200%', i) # Some G'MIC operations may reallocate the image buffer in place without risk
        i._width == i._height == 2 # Use the _width, _height, _depth, _spectrum, _data, _data_str, _is_shared read-only attributes

Args:
    numpy_array (numpy.ndarray): A non-empty 1D-4D Numpy array.
    deinterleave (Optional[bool]): If ``True``, pixel channel values will be deinterleaved inside the GmicImage data. If ``False``, pixel channels vector values will be untouched.
        Defaults to ``False``.
    permute (Optional[str]): If non-empty, a G'MIC ``permute`` operation will be run with this parameter (eg. yxzc) on the input matrix before saving into the GmicImage.
        See https://gmic.eu/reference.shtml#permute
        Defaults to "" (no permutation).

Returns:
    GmicImage: A new ``GmicImage`` based the input ``numpy.ndarray`` data.

Raises:
    GmicException, TypeError: Look at the exception message for details. Matrices with dimensions <1D or >4D will be rejected.)DOC");

PyDoc_STRVAR(
    constexpr PyGmicImage_to_numpy_helper_doc,
    R"DOC(GmicImage.to_numpy_helper(astype=numpy.float32, interleave=False, permute='', squeeze_shape=False)

Make a numpy.ndarray from a GmicImage.
G'MIC does not squeeze dimensions internally, so unless you use the ``squeeze_shape`` flag calling ``numpy.squeeze`` for you, the output matrix will be 4D.

Args:
    astype (numpy.dtype): The type to which G'MIC's float32 pixel values will cast to for the output matrix.
    interleave (Optional[bool]): If ``True``, pixel channel values will be interleaved (ie. RGB, RGB, RGB) within the numpy array. If ``False``, pixel channels vector values will be untouched/deinterleaved (ie. RRR,GGG,BBB).
        Defaults to ``False``.
    permute (Optional[str]): If non-empty, a G'MIC ``permute`` operation will be run with this parameter (eg. yxzc) on the output matrix before saving into the GmicImage.
        See https://gmic.eu/reference.shtml#permute
        Defaults to "" (ie. no permutation).

Returns:
    numpy.ndarray: A new ``numpy.ndarray`` based the input ``GmicImage`` data.)DOC");

PyDoc_STRVAR(
    constexpr PyGmicImage_to_PIL_doc,
    R"DOC(GmicImage.to_PIL(astype=numpy.uint8, squeeze_shape=True, mode='RGB')

Make a 2D 8-bit per pixel RGB PIL.Image from any GmicImage.
Equates to ``PIL.Image.fromarray(self.to_numpy_helper(astype=astype, squeeze_shape=squeeze_shape, interleave=True, permute='zyxc'), mode)``. Will import ``PIL.Image`` and ``numpy``.

This method uses ``numpy`` for conversion. Thus ``astype`` is used in a ``numpy.ndarray.astype()` conversion pass and samewise for ``squeeze_shape``.
Args:
    astype (type): Will be used for casting your image's pixel.
    squeeze_shape (bool): if True, your image shape has '1' components removed, is usually necessary to convert from G'MIC 3D to PIL.Image 2D only.
    mode (str): the PIL Image mode to use. see https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes

Returns:
    PIL.Image: A new ``PIL.Image`` based on the instance ``GmicImage`` data from which you call this method.)DOC");

PyDoc_STRVAR(constexpr PyGmicImage_from_PIL_doc,
             R"DOC(GmicImage.from_PIL(pil_image)

Make a ``GmicImage`` from a 2D ``PIL.Image.Image`` object.
Equates to ``gmic.GmicImage.from_numpy_helper(numpy.array(pil_image), deinterleave=True)``. Will import ``PIL.Image`` and ``numpy`` for conversion.


Args:
    pil_image (PIL.Image.Image): An image to convert into ``GmicImage``.

Returns:
    gmic.GmicImage: A new ``gmic.GmicImage`` based on the input ``PIL.Image.Image`` data.)DOC");

#endif

static PyObject *
PyGmicImage_copy_(const PyGmicImage *self, PyObject *)
{
    return PyObject_CallFunction(
        reinterpret_cast<PyObject *>(&PyGmicImageType), "SIIIIi",
        PyGmicImage_get_data(self, nullptr), self->_gmic_image->_width,
        self->_gmic_image->_height, self->_gmic_image->_depth,
        self->_gmic_image->_spectrum,
        static_cast<int>(self->_gmic_image->_is_shared));
}

static PyMethodDef PyGmicImage_methods[] = {
#ifdef gmic_py_numpy

    // Numpy.ndarray simplified deinterleaving Input / interleaving Output
    {"from_numpy", reinterpret_cast<PyCFunction>(PyGmicImage_from_numpy),
     METH_CLASS | METH_VARARGS | METH_KEYWORDS, PyGmicImage_from_numpy_doc},
    {"to_numpy", reinterpret_cast<PyCFunction>(PyGmicImage_to_numpy),
     METH_VARARGS | METH_KEYWORDS, PyGmicImage_to_numpy_doc},

    // Numpy.ndarray full-blown function with many helper parameters
    // Use this to build new converters
    {"from_numpy_helper",
     reinterpret_cast<PyCFunction>(PyGmicImage_from_numpy_helper),
     METH_CLASS | METH_VARARGS | METH_KEYWORDS,
     PyGmicImage_from_numpy_helper_doc},  // TODO create and set doc variable
    {"to_numpy_helper",
     reinterpret_cast<PyCFunction>(PyGmicImage_to_numpy_helper),
     METH_VARARGS | METH_KEYWORDS,
     PyGmicImage_to_numpy_helper_doc},  // TODO create and set doc variable

    // PIL (Pillow) Input / Output
    {"from_PIL", reinterpret_cast<PyCFunction>(PyGmicImage_from_PIL),
     METH_CLASS | METH_VARARGS | METH_KEYWORDS, PyGmicImage_from_PIL_doc},
    {"to_PIL", reinterpret_cast<PyCFunction>(PyGmicImage_to_PIL),
     METH_VARARGS | METH_KEYWORDS, PyGmicImage_to_PIL_doc},

    // Scikit image
    {"to_skimage", reinterpret_cast<PyCFunction>(PyGmicImage_to_skimage),
     METH_VARARGS | METH_KEYWORDS,
     PyGmicImage_to_numpy_helper_doc},  // TODO create and set doc variable
    {"from_skimage", reinterpret_cast<PyCFunction>(PyGmicImage_from_skimage),
     METH_CLASS | METH_VARARGS | METH_KEYWORDS,
     PyGmicImage_from_numpy_helper_doc},  // TODO create and set doc variable
#endif
    {"__copy__", reinterpret_cast<PyCFunction>(PyGmicImage_copy_),
     METH_VARARGS,
     "Copy method for copy.copy() support. Deepcopying and pickle-ing "
     "are not "
     "supported."},
    {nullptr} /* Sentinel */
};

static PyObject *
PyGmicImage_richcompare(PyObject *self, PyObject *other, const int op)
{
    PyObject *result;

    if (Py_TYPE(other) != Py_TYPE(self)) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    Py_INCREF(self);
    Py_INCREF(other);

    switch (op) {
        case Py_LT:
        case Py_LE:
        case Py_GT:
        case Py_GE:
            Py_XDECREF(self);
            Py_XDECREF(other);
            Py_RETURN_NOTIMPLEMENTED;
        case Py_EQ:
            // Leverage the CImg == C++ operator
            result =
                *reinterpret_cast<PyGmicImage *>(self)->_gmic_image ==
                        *reinterpret_cast<PyGmicImage *>(other)->_gmic_image
                    ? Py_True
                    : Py_False;
            break;
        case Py_NE:
            // Leverage the CImg != C++ operator
            result =
                *reinterpret_cast<PyGmicImage *>(self)->_gmic_image !=
                        *reinterpret_cast<PyGmicImage *>(other)->_gmic_image
                    ? Py_True
                    : Py_False;
            break;
        default:
            assert(false);
    }

    Py_XDECREF(self);
    Py_XDECREF(other);

    return result;
}

#ifdef cimg_use_zlib
#define zlib_enabled 1
#else
#define zlib_enabled 0
#endif

#ifdef cimg_use_png
#define libpng_enabled 1
#else
#define libpng_enabled 0
#endif

#ifdef cimg_use_tiff
#define libtiff_enabled 1
#else
#define libtiff_enabled 0
#endif

#ifdef cimg_use_jpeg
#define libjpeg_enabled 1
#else
#define libjpeg_enabled 0
#endif

#ifdef cimg_display
#define display_enabled 1
#else
#define display_enabled 0
#endif

#ifdef cimg_use_fftw3
#define fftw3_enabled 1
#else
#define fftw3_enabled 0
#endif

#ifdef cimg_use_curl
#define libcurl_enabled 1
#else
#define libcurl_enabled 0
#endif

#ifdef gmic_py_numpy
#define numpy_enabled 1
#else
#define numpy_enabled 0
#endif

#if cimg_OS == 0
#define OS_type "unknown"
#elif cimg_OS == 1
#define OS_type "unix"
#elif cimg_OS == 2
#define OS_type "windows"
#endif

PyMODINIT_FUNC
PyInit__gmic()  // NOLINT(*-reserved-identifier)
{
    // The GmicException inherits Python's builtin Exception.
    // Used for non-precise errors raised from this module.
    GmicException =
        PyErr_NewExceptionWithDoc("gmic.GmicException", /* char *name
                                                         */
                                  "Only exception class of the Gmic "
                                  "module.\n\nThis wraps G'MIC's C++ "
                                  "gmic_exception. Refer to the "
                                  "exception message itself.", /* char
                                                                * *doc
                                                                */
                                  nullptr, /* PyObject *base */
                                  nullptr /* PyObject *dict */);

    PyGmicImageType.tp_new = static_cast<newfunc>(PyGmicImage_new);
    PyGmicImageType.tp_init = nullptr;
    PyGmicImageType.tp_basicsize = sizeof(PyGmicImage);
    PyGmicImageType.tp_dealloc =
        reinterpret_cast<destructor>(PyGmicImage_dealloc);
    PyGmicImageType.tp_alloc = static_cast<allocfunc>(PyGmicImage_alloc);
    PyGmicImageType.tp_free = PyObject_Free;
    PyGmicImageType.tp_methods = PyGmicImage_methods;
    PyGmicImageType.tp_repr = reinterpret_cast<reprfunc>(PyGmicImage_repr);
    PyGmicImageType.tp_call = static_cast<ternaryfunc>(PyGmicImage_call);
    PyGmicImageType.tp_getattro = PyObject_GenericGetAttr;
    PyGmicImageType.tp_doc = PyGmicImage_doc;
    PyGmicImageType.tp_members = nullptr;
    PyGmicImageType.tp_getset = PyGmicImage_getsets;
    PyGmicImageType.tp_richcompare = PyGmicImage_richcompare;
    PyGmicImageType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;

    if (PyType_Ready(&PyGmicImageType) < 0)
        return nullptr;

    PyGmicType.tp_new = PyGmic_new;
    PyGmicType.tp_basicsize = sizeof(PyGmic);
    PyGmicType.tp_methods = PyGmic_methods;
    PyGmicType.tp_repr = reinterpret_cast<reprfunc>(PyGmic_repr);
    PyGmicType.tp_init = nullptr;
    PyGmicType.tp_alloc = PyGmic_alloc;
    PyGmicType.tp_getattro = PyObject_GenericGetAttr;
    PyGmicType.tp_dealloc = reinterpret_cast<destructor>(PyGmic_dealloc);
    PyGmicType.tp_free = PyObject_Free;
    PyGmicType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;

    if (PyType_Ready(&PyGmicType) < 0)
        return nullptr;

    PyObject *m = PyModule_Create(&gmic_module);
    if (m == nullptr) {
        return nullptr;
    }
#ifdef gmic_py_jupyter_ipython_display
    autoload_wurlitzer_into_ipython();
#endif

    Py_INCREF(&PyGmicImageType);
    Py_INCREF(&PyGmicType);
    Py_INCREF(GmicException);
    PyModule_AddObject(m, "GmicImage",
                       reinterpret_cast<PyObject *>(
                           &PyGmicImageType));  // Add GmicImage object
                                                // to the module
    PyModule_AddObject(m, "Gmic",
                       reinterpret_cast<PyObject *>(
                           &PyGmicType));  // Add Gmic object to the module
    PyModule_AddObject(m, "GmicException",
                       GmicException);  // Add Gmic object to the module
    PyModule_AddObject(
        m, "__version__",
        PyUnicode_Join(PyUnicode_FromString("."),
                       PyUnicode_FromFormat("%d", gmic_version)));
    PyModule_AddObject(
        m, "__build__",
        PyUnicode_FromFormat(
            "zlib_enabled:%d libpng_enabled:%d libtiff_enabled:%d "
            "libjpeg_enabled:%d display_enabled:%d "
            "fftw3_enabled:%d libcurl_enabled:%d openmp_enabled:%d cimg_OS:%d "
            "numpy_enabled:%d "
            "OS_type:%s",
            zlib_enabled, libpng_enabled, libtiff_enabled, libjpeg_enabled,
            display_enabled, fftw3_enabled, libcurl_enabled, cimg_use_openmp,
            cimg_OS, numpy_enabled, OS_type));
    // For more debugging, the user can look at __spec__ automatically
    // set by setup.py
    Py_XDECREF(&PyGmicImageType);
    Py_XDECREF(&PyGmicType);
    Py_XDECREF(GmicException);

    return m;
}
#undef OS_type
#undef display_enabled
#undef fftw3_enabled
#undef libcurl_enabled
#undef libjpeg_enabled
#undef libpng_enabled
#undef libtiff_enabled
#undef numpy_enabled
#undef zlib_enabled

try:
    import gmic

    if '__build__' not in dir(gmic):
        raise ImportError("Wrong gmic module imported: no __build__ attribute (%s)" % gmic)
except ImportError as err:
    import sys

    err.msg += "\nsys.path = %s" % sys.path
    raise

# Tests parametrization: run calls to gmic.run() and gmic.Gmic().run() should have the same behaviour!
gmic_instance_types = {
    "argnames": "gmic_instance_run",
    "argvalues": [gmic.run, gmic.Gmic().run],
    "ids": ["gmic_module_run", "gmic_instance_run"],
}

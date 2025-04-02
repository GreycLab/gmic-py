ARG IMGFROM
FROM ${IMGFROM}
ARG IMGFROM
LABEL org.opencontainers.image.source="https://github.com/GreycLab/gmic-py/"
LABEL org.opencontainers.image.descriptio="gmic-py ${IMGFROM##*/} build image"
LABEL org.opencontainers.image.licenses="CECILL-2.1"

COPY before_all.sh /tmp
RUN --mount=type=cache,target=/var/cache/dnf/ \
    --mount=type=cache,target=/var/cache/apk/ \
     /tmp/before_all.sh

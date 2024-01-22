docker run -d --privileged -it --env=LOCAL_USER_ID="$(id -u)" \
--gpus 'all,"capabilities=display,video,utility,graphics,compute,compat32"' \
-e DISPLAY=$DISPLAY -e color_prompt=yes \
-v /home/wjh/code_ws/POMDP:/src/POMDP:rw \
-v /tmp/.X11-unix:/tmp/.X11-unix:ro \
-p 40400:22 \
--network bridge \
--name=mkl veijh/mkl:dev \
/bin/bash

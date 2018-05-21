This is a semiautomated Dockerfile, which sets up a build environment for
building Tensorflow wheels and running tests.  Typing make will display help,
and building the enviornment is done by running:

make build

prompts will be given to either run code within this docker container or to build
a tensorflow wheel within docker: tfbuild doall.

typing "make" outside the container or "tfbuild" from inside the container will
display a help menu.

To detach from within the container, type CONTROL-P, CONTROL-Q.  To reenter that
container, type make attach.

---

To list the current executing containers, it's recommended to add this code
snippet to the end of your .bashrc file:

# docker aliases
alias dcls='docker container ls -a'
alias dils='docker image ls -a'
alias dvls='docker volume ls -a'
alias dclean='docker system prune -a'

dcls - list active and inactive containers
dils - list active and inactive images
dvls - list active and inactive volumes
dclean - remove all inactive assets

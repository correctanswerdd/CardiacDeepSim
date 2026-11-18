# Builds a Docker image with OpenTidalFarm master
# version built from gitsources. It is based on
# the dev-dolfin-adjoint image available at
#
#      quay.io/dolfinadjoint/dev-dolfin-adjoint
#
# Authors:
# Simon Funke <simon@simula.no>

FROM quay.io/fenicsproject/dev:latest
MAINTAINER Simon Funke <simon@simula.no>

USER root

RUN apt-get update && apt-get install -y \
    mercurial

RUN git clone https://github.com/ComputationalPhysiology/cbcbeat.git
RUN cd cbcbeat; pip install .

USER fenics
COPY WELCOME $FENICS_HOME/WELCOME

USER root

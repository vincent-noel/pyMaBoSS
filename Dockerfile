FROM colomoto/colomoto-docker
EXPOSE 8888


RUN pip install -U pyparsing \
    && pip install -U pandas \
    && pip install -U matplotlib

RUN pip install -U --user git+https://github.com/GINsim/GINsim-python

RUN conda install -c conda-forge ipywidgets --yes

RUN pip install -U --user git+https://github.com/thenlevy/pyMaBoSS
COPY model /model
COPY notebook /notebook/

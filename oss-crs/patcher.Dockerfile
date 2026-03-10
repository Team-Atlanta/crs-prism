# =============================================================================
# crs-prism Patcher Module
# =============================================================================
# RUN phase: Receives POVs, generates patches using Prism,
# tests them using the snapshot image for incremental rebuilds.
#
# Uses host Docker socket (mounted by framework) to access snapshot images.
# =============================================================================

# These ARGs are required by the oss-crs framework template
ARG target_base_image
ARG crs_version

FROM prism-base

# Install libCRS (CLI + Python package)
COPY --from=libcrs . /libCRS
RUN pip3 install /libCRS \
    && python3 -c "from libCRS.base import DataType; print('libCRS OK')"

# Install crs-prism package (patcher + agents + crete)
COPY pyproject.toml /opt/crs-prism/pyproject.toml
COPY patcher.py /opt/crs-prism/patcher.py
COPY agents/ /opt/crs-prism/agents/
COPY crete/ /opt/crs-prism/crete/
RUN pip3 install /opt/crs-prism

CMD ["run_patcher"]

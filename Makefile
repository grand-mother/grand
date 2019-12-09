PREFIX=     $(PWD)
BUILD_TYPE= release

PYTHON= $(PWD)/bin/python
INSTALL_X= install -m 0755
INSTALL_F= install -m 0644

BUILD_DIR= $(PWD)/build-$(BUILD_TYPE)

ifeq ($(BUILD_TYPE), release)
CFLAGS+= -O3
else ifeq ($(BUILD_TYPE), debug)
CFLAGS+= -O0 -g3
else
$(error invalid BUILD_TYPE $(BUILD_TYPE))
endif

LIBS= turtle gull
GULL_DATA= IGRF12.COF WMM2015.COF

INSTALL_LIBS= $(addprefix $(PREFIX)/lib/lib,$(addsuffix .so,$(LIBS)))
INSTALL_DATA= $(addprefix $(PREFIX)/lib/python/grand/libs/data/gull/,$(GULL_DATA))

BUILD_LIBS=   $(BUILD_DIR)/lib/python/grand/_core.so
BUILD_LIBS+=  $(addprefix $(BUILD_DIR)/lib/lib,$(addsuffix .so,$(LIBS)))

MODULES=  $(PREFIX)/lib/python/grand/_core.so
MODULES+= $(addprefix $(PREFIX)/,$(shell find lib/python/grand -name *.py))

$(BUILD_LIBS) $(addprefix $(BUILD_DIR)/src/gull/share/data/,$(GULL_DATA)): FORCE
	@echo "==== Building $$(basename $@) module ===="
	@$(MAKE) -C src BUILD_DIR=$(BUILD_DIR) PYTHON=$(PYTHON) CFLAGS=$(CFLAGS)
	@echo "==== Successfully built $$(basename $@) module ===="

FORCE:

install: $(INSTALL_LIBS) $(MODULES) $(INSTALL_DATA)
	@echo "==== Successfully installed libraries and modules ===="

$(PREFIX)/%.so: $(BUILD_DIR)/%.so
	@echo "INSTALL  $$(basename $@)" && \
	$(INSTALL_X) -D $^ $@

$(PREFIX)/lib/python/%.py: lib/python/%.py
	@echo "INSTALL  $$(basename $@)" && \
	$(INSTALL_X) -D $^ $@

$(PREFIX)/lib/python/grand/libs/data/gull/%: $(BUILD_DIR)/src/gull/share/data/%
	@echo "INSTALL  $$(basename $@)" && \
	$(INSTALL_F) -D $^ $@

clean:
	@$(RM) -rf lib/*.so lib/python/grand/*.so lib/python/grand/libs/data build-*

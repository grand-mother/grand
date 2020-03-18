PREFIX=     $(PWD)
BUILD_TYPE= release

PYTHON= $(PWD)/bin/python
INSTALL_X= install -m 0755
INSTALL_F= install -m 0644
INSTALL_D= install -d

BUILD_DIR= $(PWD)/build-$(BUILD_TYPE)

ifeq ($(BUILD_TYPE), release)
CFLAGS+= -O3
else ifeq ($(BUILD_TYPE), debug)
CFLAGS+= -O0 -g3
else
$(error invalid BUILD_TYPE $(BUILD_TYPE))
endif

SOEXT= so
SYS  = $(shell uname -s)
ifeq ($(SYS), Darwin)
	SOEXT= dylib
endif

LIBS= turtle gull
GULL_DATA= IGRF12.COF WMM2015.COF

INSTALL_LIBS= $(addprefix $(PREFIX)/grand/libs/lib,$(addsuffix .$(SOEXT),$(LIBS)))
INSTALL_DATA= $(addprefix $(PREFIX)/grand/libs/data/gull/,$(GULL_DATA))

BUILD_LIBS=   $(BUILD_DIR)/grand/_core.so
BUILD_LIBS+=  $(addprefix $(BUILD_DIR)/grand/libs/lib,$(addsuffix .$(SOEXT),$(LIBS)))

MODULES=  $(PREFIX)/grand/_core.so
MODULES+= $(addprefix $(PREFIX)/,$(shell find grand -name *.py 2>/dev/null))

$(BUILD_LIBS) $(addprefix $(BUILD_DIR)/src/gull/share/data/,$(GULL_DATA)): FORCE
	@echo "==== Building $$(basename $@) module ===="
	@$(MAKE) -C src BUILD_DIR=$(BUILD_DIR) PYTHON=$(PYTHON) CFLAGS=$(CFLAGS)
	@echo "==== Successfully built $$(basename $@) module ===="

FORCE:

install: $(INSTALL_LIBS) $(MODULES) $(INSTALL_DATA)
	@echo "==== Successfully installed libraries and modules ===="

$(PREFIX)/%.so: $(BUILD_DIR)/%.so
	@echo "INSTALL  $$(basename $@)" && \
	$(INSTALL_D) $(shell dirname $@) && \
	$(INSTALL_X) $^ $@

$(PREFIX)/%.dylib: $(BUILD_DIR)/%.dylib
	@echo "INSTALL  $$(basename $@)" && \
	$(INSTALL_D) $(shell dirname $@) && \
	$(INSTALL_X) $^ $@

$(PREFIX)/%.py: %.py
	@echo "INSTALL  $$(basename $@)" && \
	$(INSTALL_D) $(shell dirname $@) && \
	$(INSTALL_X) $^ $@

$(PREFIX)/grand/libs/data/gull/%: $(BUILD_DIR)/src/gull/share/data/%
	@echo "INSTALL  $$(basename $@)" && \
	$(INSTALL_D) $(shell dirname $@) && \
	$(INSTALL_F) $^ $@

clean:
	@$(RM) -r grand/*.so grand/libs/*.$(SOEXT) grand/libs/data build* dist* *.egg-info
	@find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf

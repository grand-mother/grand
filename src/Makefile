SYS= $(shell uname -s)

CC=      gcc
CFLAGS=  -O3
LDFLAGS= '-Wl,-rpath,$$$$ORIGIN'
SOEXT=   so
ifeq ($(SYS), Darwin)
	LDFLAGS= -Wl,-rpath,@loader_path
	SOEXT=   dylib
endif
export LDFLAGS
INSTALL_X= install -m 0755
INSTALL_F= install -m 0644
INSTALL_D= install -d
PYTHON   = python3

BUILD_DIR= build

# Git tags of external dependencies
TURTLE_TAG= v0.8
GULL_TAG  = 286ace5

define git_clone #([$1] = destination, [$2] = url, [$3] = tag)
	echo "CLONE    $$(basename $1)" && \
	git clone -q "$2" "$1" && \
	cd "$1" && \
	git checkout -q "$3"
endef

define build_lib #([$1] = name, [$2] = source, [$3] = library)
	echo "MAKE     $$(basename $(3))" && \
	$(MAKE) --quiet -C $(2) CFLAGS="$(CFLAGS) -DTURTLE_NO_TIFF -std=c99 -pedantic" TURTLE_USE_TIFF=0 && \
	$(INSTALL_D) $(shell dirname $(3)) && \
	$(INSTALL_F) $(2)/lib/lib$(1).$(SOEXT) $(3)
endef

define install_header #([$1] = name, [$2] = source, [$3] = header)
	echo "INSTALL  $$(basename $(3))" && \
	$(INSTALL_D) $(shell dirname $(3)) && \
	$(INSTALL_F) $(2)/include/$(1).h $(3)
endef

LIBS= $(BUILD_DIR)/grand/libs/libturtle.$(SOEXT) \
      $(BUILD_DIR)/grand/libs/libgull.$(SOEXT)

MODULES= $(BUILD_DIR)/grand/_core.so

HEADERS= $(BUILD_DIR)/include/turtle.h \
         $(BUILD_DIR)/include/gull.h

all: $(LIBS) $(HEADERS) $(MODULES)

.PHONY: clean

clean:
	$(RM) -rf $(BUILD_DIR)

$(BUILD_DIR)/src/turtle:
	@$(call git_clone,$@,"https://github.com/niess/turtle.git",$(TURTLE_TAG))

$(BUILD_DIR)/src/gull:
	@$(call git_clone,$@,"https://github.com/niess/gull.git",$(GULL_TAG))

$(BUILD_DIR)/grand/libs/lib%.$(SOEXT): $(BUILD_DIR)/src/%
	@$(call build_lib,$*,$^,$@)

$(BUILD_DIR)/include/%.h: $(BUILD_DIR)/src/%
	@$(call install_header,$*,$^,$@)

$(MODULES): $(LIBS) $(HEADERS) grand.c grand.h
	@echo "PYTHON   $$(basename $@)" && \
	$(PYTHON) make.py $(BUILD_DIR)

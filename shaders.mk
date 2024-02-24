SCR_DIR := ./
BUILD_DIR := ./build

vpath %.spv $(BUILD_DIR)

SRCS := $(wildcard $(SRC_DIR)*.frag $(SRC_DIR)*.vert)

OBJS := $(SRCS:%.frag=$(BUILD_DIR)/%Fragment.spv)
OBJS := $(OBJS:%.vert=$(BUILD_DIR)/%Vertex.spv)

COMPILE = $(shell mkdir -p $(BUILD_DIR) && glslc -o $(2) $(1))

#COMPILE = $(shell mkdir -p $(dir $(2)) && glslc -o $(2) $(1))
#COMPILE = 0:$(0) 1:$(1) 2:$(2)


entry: $(OBJS)
	@echo SRCS: $(SRCS)
	@echo OBJS: $(OBJS)

$(BUILD_DIR)/%Vertex.spv: %.vert
	$(call COMPILE,$<,$@)

$(BUILD_DIR)/%Fragment.spv: %.frag
	$(call COMPILE,$<,$@)

.PHONY: entry

vpath %Vertex.spv ./debug
vpath %Fragment.spv ./debug

SRC := $(wildcard ../*.frag ../*.vert)
OUT1 := $(SRC:.vert=Vertex.spv)
OUT2 := $(OUT1:.frag=Fragment.spv)

all: ./debug $(OUT2)
	echo $(SRC)
.PHONY = all

./debug:
	mkdir -p debug

%Vertex.spv: %.vert
	$(shell glslc -o ./debug/$@ $<)

%Fragment.spv: %.frag
	$(shell glslc -o ./debug/$@ $<)

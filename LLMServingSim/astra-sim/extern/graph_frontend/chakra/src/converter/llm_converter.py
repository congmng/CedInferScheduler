import logging
from io import TextIOWrapper
from typing import Any, List
from enum import Enum

from ...schema.protobuf.et_def_pb2 import *
from ...schema.protobuf.et_def_pb2 import AttributeProto as ChakraAttr
from ..third_party.utils.protolib import encodeMessage as encode_message


# Memory type is deprecated for latest version of chakra & astra-sim
# It only uses tensor_size for the remote memory issue in Workload.cc
# Added memory types using enum and add tensor_loc in node attribute in et_feeder_node.cpp
class MemoryType(Enum):
    INVALID_MEMORY = 0
    LOCAL_MEMORY = 1
    REMOTE_MEMORY = 2
    CXL_MEMORY = 3
    STORAGE_MEMORY = 4


class Layer:
    def __init__(self, line: str = None, cols: List[str] = None):
        try:
            # ``cols`` lets a caller that already holds the fields skip the
            # format-and-resplit round trip; ``line`` is the text path.
            col = cols if cols is not None else line.strip().split()
            if col[0] == 'EXPERT': # If Expert Flag
                self.name = col[0]
                self.expert_num = col[1]
                self.comm_type, self.involved_dim = self._parse_comm_type(str(col[2]) if len(col) > 2 else "NONE")
                self.comm_size = int(col[3]) if len(col) > 3 else 0
                self.is_expert = True
                self.is_pim = False
                self.comm_node = None
                self.comp_node = None
            elif col[0] == 'PIM': # If PIM Flag
                self.name = col[0]
                self.pim_num = col[1]
                self.is_expert = False
                self.is_pim = True
                self.comm_node = None
                self.comp_node = None
                self.comm_type = "NONE"
                self.involved_dim = None
            else:
                self.is_expert = False
                self.is_pim = False
                self.name = col[0]

                # compuation
                self.comp_time = int(col[1])
                self.comp_node = None

                # memory
                self.input_memory_loc = str(col[2])
                self.input_memory_size = int(col[3])
                self.input_memory_node = None
                self.weight_memory_loc = str(col[4])
                self.weight_memory_size = int(col[5])
                self.weight_memory_node = None
                self.output_memory_loc = str(col[6])
                self.output_memory_size = int(col[7])
                self.output_memory_node = None

                # communication (supports ALLREDUCE:1,0 format for involved_dim)
                self.comm_type, self.involved_dim = self._parse_comm_type(str(col[8]))
                self.comm_size = int(col[9])
                self.comm_node = None

                self.misc = str(col[10])
        except:
            raise ValueError(f"Cannot parse the following layer -- \"{line}\"")

    @staticmethod
    def _parse_comm_type(s: str):
        """Parse comm_type string, extracting optional involved_dim.

        'ALLREDUCE:1,0' -> ('ALLREDUCE', [True, False])
        'ALLTOALL'      -> ('ALLTOALL', None)
        'NONE'          -> ('NONE', None)
        """
        if ':' in s:
            comm_type, dim_str = s.split(':', 1)
            involved_dim = [v == '1' for v in dim_str.split(',')]
            return comm_type, involved_dim
        return s, None
        
class LLMConverter:
    def __init__(
        self,
        input_filename: str,
        output_filename: str,
        num_npus: int,
        npu_offset: int = 0,
        local_offloading: bool = False,
    ):
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.num_npus = num_npus
        self.npu_offset = npu_offset
        self.local_offloading = local_offloading
        self.next_node_id = 0

        # For send & recv nodes
        self.next_comm_tag = 0
        self.comm_tag_dict = dict()

        # Set by convert_rows so get_layers can return them instead of
        # parsing a file. See convert_rows.
        self._layers = None

    def get_global_metadata(self):
        # ``input_file`` carries the trace's *path*, not its contents.
        #
        # It used to embed the whole trace text. Nothing consumes it:
        # ETFeeder::readGlobalMetadata() reads the message into a local
        # shared_ptr and drops it on the floor, and no other reader touches
        # the attribute. So every byte of it was written by the converter,
        # re-parsed by the feeder's protobuf reader, and discarded.
        #
        # It was not a small overhead. On the swe-bench MoE DP+EP example a
        # trace is ~82 KB of a 117 KB .et -- 70% of the file -- and the
        # simulator generates 8,810 of them for one session, so it was
        # ~720 MB of encode/write/read/parse per run, plus a second full
        # read of the trace file here purely to obtain the text.
        #
        # The path keeps the provenance that made this attribute useful for
        # debugging; the trace itself is written to disk by --save-trace-text.
        attr = [
            ChakraAttr(name="schema", string_val="1.0.2-chakra.0.0.4"),
            ChakraAttr(name="input_file", string_val=self.input_filename),
        ]
        metadata = GlobalMetadata(attr=attr)
        return metadata
    
    def get_layers(self, f: TextIOWrapper) -> List[Layer]:
        if self._layers is not None:
            return self._layers
        layers: List[Layer] = []
        for line in f:
            layers.append(Layer(line))
        return layers

    def convert_rows(self, header_line: str, rows: List[List[str]]) -> None:
        """Convert from pre-parsed field lists, with no text round trip.

        The simulator already holds every field. Formatting them into padded
        columns, writing the file, reading it back and splitting each line
        again is pure overhead now that the converter runs in the same
        process -- 0.65 ms of formatting and writing per batch plus 0.9 ms of
        reading and re-parsing, against 8,810 batches on the swe-bench MoE
        DP+EP example.

        Nothing downstream changes: convert_common, convert_prefill and
        convert_event each touch the file handle exactly once, to call
        get_layers, so seeding the layers is the whole of it. Header parsing
        mirrors convert() on a string instead of a readline, and num_layers
        is len(rows) -- which is precisely what convert() reads off line two,
        since that is what the writer puts there.
        """
        self._layers = [Layer(cols=cols) for cols in rows]

        first_line = header_line.strip().split()
        execution_type = first_line[0]

        header = {}
        fields = first_line[1:]
        for i in range(0, len(fields) - 1, 2):
            if fields[i].endswith(":"):
                header[fields[i][:-1]] = fields[i + 1]

        num_npu_group = int(header.get("model_parallel_NPU_group", 0))
        boundary_str = header.get("pp_stage_boundaries", "")
        stage_boundaries = (
            [int(b) for b in boundary_str.split(",")] if boundary_str else []
        )
        num_layers = len(rows)

        if execution_type in ("COLOCATED", "DECODE"):
            if num_npu_group <= 0:
                raise ValueError(f"model_parallel_NPU_group <= 0")
            self.convert_common(None, num_layers, num_npu_group, stage_boundaries)
        elif execution_type == "PREFILL":
            if num_npu_group <= 0:
                raise ValueError(f"model_parallel_NPU_group <= 0")
            target = int(header["pd_decode_npu_offset"]) if "pd_decode_npu_offset" in header else None
            self.convert_prefill(None, num_layers, num_npu_group, stage_boundaries, target)
        elif execution_type == "EVENT":
            self.convert_event(None, num_layers)
        else:
            raise ValueError(f"Unsupported execution type, {execution_type}")

    def get_next_node_id(self) -> int:
        ret = self.next_node_id
        self.next_node_id += 1
        return ret
    
    def get_next_comm_tag(self) -> int:
        ret = self.next_comm_tag
        self.next_comm_tag += 1
        return ret

    def get_node(self, name: str, node_type: NodeType) -> Any:
        node = Node()
        node.id = self.get_next_node_id()
        node.name = name
        node.type = node_type
        return node

    def get_comp_node(self, layer_name: str, comp_time: int) -> Any:
        node = self.get_node("COMP_NODE_" + layer_name, COMP_NODE)
        node.duration_micros = comp_time
        return node
    
    def get_comm_type(self, comm_type: str) -> int:
        if comm_type == "ALLREDUCE":
            return ALL_REDUCE
        elif comm_type == "ALLTOALL":
            return ALL_TO_ALL
        elif comm_type == "ALLGATHER":
            return ALL_GATHER
        elif comm_type == "REDUCESCATTER":
            return REDUCE_SCATTER
        return 0
    
    def get_comm_coll_node(self, layer_name: str, comm_type: str, comm_size: int,
                           involved_dim: list = None) -> Any:
        node = self.get_node(f"COMM_COLL_NODE_{layer_name}_{comm_type}", COMM_COLL_NODE)
        node.attr.append(ChakraAttr(name="comm_type", int64_val=self.get_comm_type(comm_type)))
        node.attr.append(ChakraAttr(name="comm_size", int64_val=comm_size))
        if involved_dim is not None:
            node.attr.append(ChakraAttr(name="involved_dim",
                                        bool_list=BoolList(values=involved_dim)))
        return node

    def get_comm_node(self, is_send: bool, layer_name: str, comm_type: str, comm_size: int,
                           comm_src: int, comm_dst: int, id: int = 0) -> Any:
        if is_send:
            node = self.get_node(
                    f"COMM_SEND_NODE_{layer_name}_{comm_type}_{comm_src}_{comm_dst}",
                    COMM_SEND_NODE)
        else:
            node = self.get_node(
                    f"COMM_RECV_NODE_{layer_name}_{comm_type}_{comm_src}_{comm_dst}",
                    COMM_RECV_NODE)
        node.attr.append(ChakraAttr(name="comm_type", int64_val=self.get_comm_type(comm_type)))
        node.attr.append(ChakraAttr(name="comm_src", int32_val=comm_src))
        node.attr.append(ChakraAttr(name="comm_dst", int32_val=comm_dst))
        node.attr.append(ChakraAttr(name="comm_size", int64_val=comm_size))
        comm_key = f"{comm_src}_{comm_dst}_{id}"
        if comm_key in self.comm_tag_dict:
            node.attr.append(ChakraAttr(name="comm_tag", int32_val=self.comm_tag_dict[comm_key]))
        else:
            new_tag = self.get_next_comm_tag()
            node.attr.append(ChakraAttr(name="comm_tag", int32_val=new_tag))
            self.comm_tag_dict[comm_key] = new_tag

        # check if SEND/RECV pair have same tags
        # print(f"name: {node.name}, src: {node.comm_src}, dst: {node.comm_dst}, size: {node.comm_size}, key: {comm_key}, tag: {node.comm_tag}")
        return node
    
    def get_mem_type(self, mem_type: str) -> int:
        mem_type = mem_type.split(':')[0]  # Exclude the device number if present
        if mem_type == "LOCAL":
            return MemoryType.LOCAL_MEMORY.value
        elif mem_type == "REMOTE":
            return MemoryType.REMOTE_MEMORY.value
        elif mem_type == "CXL":
            return MemoryType.CXL_MEMORY.value
        elif mem_type == "STORAGE":
            return MemoryType.STORAGE_MEMORY.value
        return MemoryType.INVALID_MEMORY.value
    
    def get_mem_device(self, mem_type: str) -> int:
        """
        Extract the device index from mem_type.

        Supported formats:
        - "REMOTE"          -> returns 0
        - "REMOTE:1"        -> returns 1
        - "REMOTE:1.3"      -> returns 1  (device = 1, channel = 3)
        """
        parts = mem_type.split(":", 1)
        if len(parts) == 2:
            # parts[1] may be "1" or "1.3"
            dev_part = parts[1].split(".", 1)[0]
            if dev_part.isdigit():
                return int(dev_part)
        return 0  # Default device number


    def get_mem_channel(self, mem_type: str) -> int:
        """
        Extract the channel index from mem_type.

        Supported formats:
        - "REMOTE"          -> returns 0
        - "REMOTE:1"        -> returns 0  (no channel specified)
        - "REMOTE:1.3"      -> returns 3  (channel = 3)
        """
        parts = mem_type.split(":", 1)
        if len(parts) == 2:
            # parts[1] may be "1" or "1.3"
            sub = parts[1].split(".", 1)
            if len(sub) == 2:
                chan_part = sub[1]
                if chan_part.isdigit():
                    return int(chan_part)
        return 0  # Default channel number


    def get_memory_load_node(self, layer_name: str, tensor_type: str, mem_type: str, tensor_size: int) -> Any:
        node = self.get_node("MEM_LOAD_NODE_" + layer_name + "_" + tensor_type, MEM_LOAD_NODE)
        node.attr.append(ChakraAttr(name="tensor_size", uint64_val=tensor_size))
        node.attr.append(ChakraAttr(name="tensor_loc", uint32_val=self.get_mem_type(mem_type)))
        node.attr.append(ChakraAttr(name="tensor_device", uint32_val=self.get_mem_device(mem_type)))
        return node

    def get_memory_store_node(self, layer_name: str, tensor_type: str, mem_type: str, tensor_size: int) -> Any:
        node = self.get_node("MEM_STORE_NODE_" + layer_name + "_" + tensor_type, MEM_STORE_NODE)
        node.attr.append(ChakraAttr(name="tensor_size", uint64_val=tensor_size))
        node.attr.append(ChakraAttr(name="tensor_loc", uint32_val=self.get_mem_type(mem_type)))
        node.attr.append(ChakraAttr(name="tensor_device", uint32_val=self.get_mem_device(mem_type)))
        return node

    def get_pim_compute_node(self, layer_name: str, tensor_type: str, comp_time: int, mem_type: str, tensor_size: int) -> Any:
        node = self.get_node("PIM_COMP_NODE_" + layer_name + "_" + tensor_type, PIM_COMP_NODE)
        node.duration_micros = comp_time
        node.attr.append(ChakraAttr(name="tensor_size", uint64_val=tensor_size))
        node.attr.append(ChakraAttr(name="tensor_loc", uint32_val=self.get_mem_type(mem_type)))
        node.attr.append(ChakraAttr(name="tensor_device", uint32_val=self.get_mem_device(mem_type)))
        node.attr.append(ChakraAttr(name="tensor_channel", uint32_val=self.get_mem_channel(mem_type)))
        return node
        
    def add_parent(self, child_node: Any, parent_node: Any) -> None:
        child_node.data_deps.append(parent_node.id)

    def get_stage_edges(self, num_layers: int, num_npu_group: int,
                        stage_boundaries: List[int]) -> List[Any]:
        """Resolve the [start, end) trace-line range owned by each pipeline stage.

        ``stage_boundaries`` comes from the trace header and holds the line
        index at which every stage after the first begins. The frontend puts
        those cuts on transformer-block boundaries, which is what makes the
        tensor crossing a stage boundary the hidden state: the sending
        layer's output_size then equals the receiving layer's input_size, and
        the SEND/RECV pair matches in ASTRA-sim (its callback tracker keys on
        chunk size, so a mismatch deadlocks the run instead of erroring).
        Splitting the line count evenly instead lands cuts inside a block --
        e.g. between qkv_proj and rotary_emb, whose declared sizes differ by
        the V projection -- so do not reintroduce that.
        """
        if num_npu_group == 1:
            return [(0, num_layers)]
        if len(stage_boundaries) != num_npu_group - 1:
            raise ValueError(
                f"trace declares model_parallel_NPU_group: {num_npu_group} but "
                f"carries {len(stage_boundaries)} pp_stage_boundaries "
                f"(expected {num_npu_group - 1}); regenerate the trace")
        edges = [0] + list(stage_boundaries) + [num_layers]
        for i in range(len(edges) - 1):
            if not 0 <= edges[i] < edges[i + 1] <= num_layers:
                raise ValueError(
                    f"pp_stage_boundaries {stage_boundaries} are not a strictly "
                    f"increasing split of {num_layers} layers")
        return [(edges[i], edges[i + 1]) for i in range(num_npu_group)]

    def convert_common(self, f: TextIOWrapper, num_layers: int, num_npu_group: int,
                       stage_boundaries: List[int] = None):
        layers: list[Layer] = self.get_layers(f)

        # vllm: check eviction or load
        evict = None
        load = None
        ev_ld_cnt = 0
        for i in range(2):
            if 'kv_load' in layers[i].name:
                load = self.get_memory_load_node(
                            layers[i].name,
                            "WEIGHT",
                            layers[i].weight_memory_loc,
                            layers[i].weight_memory_size, # already per npu kv_cache size
                        )
            elif 'kv_evict' in layers[i].name:
                evict = self.get_memory_store_node(
                            layers[i].name,
                            "WEIGHT",
                            layers[i].weight_memory_loc,
                            layers[i].weight_memory_size,
                        )
            else:
                continue
            ev_ld_cnt += 1
            
        layers = layers[ev_ld_cnt:]
        num_layers -= ev_ld_cnt

        if self.num_npus % num_npu_group != 0: print("Warning! num_npus % num_npu_group != 0, Some npus won't do anything!")
        npus_per_group = self.num_npus // num_npu_group
        if npus_per_group == 1: # same as pipeline parallelism, ignore all reduce
            use_comm = False
        else:
            use_comm = True
        stage_edges = self.get_stage_edges(num_layers, num_npu_group,
                                           stage_boundaries or [])

        for npu_group in range(num_npu_group):
            for npu_offset in range(npus_per_group):
                # Re-read the authoritative edges per rank: the walk below may
                # rebind layer_end if it ever overruns.
                layer_start, layer_end = stage_edges[npu_group]
                npu_id = npu_group * npus_per_group + npu_offset + self.npu_offset
                output_filename = "%s.%d.et" % (self.output_filename, npu_id)
                first_comp_node = True
                with open(output_filename, "wb") as g:
                    global_metadata = self.get_global_metadata()
                    encode_message(g, global_metadata)
                    if evict != None:
                        encode_message(g, evict)
                    if load != None:
                        encode_message(g, load)
                    if npu_group == 0:
                        # Load Input
                        input_load_node = self.get_memory_load_node(
                            layers[layer_start].name,
                            "INPUT",
                            layers[layer_start].input_memory_loc,
                            layers[layer_start].input_memory_size,
                        )
                        encode_message(g, input_load_node)                  
                    else:
                        if layers[layer_start].is_expert or layers[layer_start].is_pim:
                            # Receive input (from the previous layer in another npu group)
                            receive_input_node = self.get_comm_node(
                                is_send=False,
                                layer_name=layers[layer_start-1].name,
                                comm_type=layers[layer_start-1].comm_type,
                                comm_size=layers[layer_start-1].output_memory_size,
                                comm_src=npu_id - npus_per_group,
                                comm_dst=npu_id
                            )
                            encode_message(g, receive_input_node)
                        else:
                            # Receive input (from the previous layer in another npu group)
                            receive_input_node = self.get_comm_node(
                                is_send=False,
                                layer_name=layers[layer_start].name,
                                comm_type=layers[layer_start].comm_type,
                                comm_size=layers[layer_start].input_memory_size,
                                comm_src=npu_id - npus_per_group,
                                comm_dst=npu_id
                            )
                            encode_message(g, receive_input_node)

                    expert_start = False
                    pim_start = False
                    attn_remain = False # to handle remaining prefill attention after pim
                    pim_parent_nodes = []
                    pim_comp_nodes = []
                    past_pim_comp_nodes = []
                    last_batch_type = "BATCH_1"
                    layer_num = layer_start
                    while expert_start or pim_start or attn_remain or layer_num < layer_end:
                        if not layers[layer_num].is_expert and not layers[layer_num].is_pim: 
                            if (self.local_offloading or layers[layer_num].weight_memory_loc != "LOCAL") and layers[layer_num].weight_memory_size > 0:
                                # Load weight (for weight offloading)
                                weight_load_node = self.get_memory_load_node(
                                    layers[layer_num].name,
                                    "WEIGHT",
                                    layers[layer_num].weight_memory_loc,
                                    layers[layer_num].weight_memory_size,
                                )
                                layers[layer_num].weight_memory_node = weight_load_node
                                if expert_start:
                                    self.add_parent(weight_load_node, comp_node) # dependent to previous comp_node due to gate function
                                encode_message(g, weight_load_node)
                            # Compute
                            if layers[layer_num].comp_time != 0 and not pim_start: # pim computation is handled pim_comp_node
                                comp_node = self.get_comp_node(
                                    layers[layer_num].name, 
                                    layers[layer_num].comp_time)
                                layers[layer_num].comp_node = comp_node

                                # handle pim parent nodes, and if prefill attention remains wait until all attention is done (before o_proj)
                                if len(pim_parent_nodes) != 0:
                                    if attn_remain:
                                        for parent in pim_parent_nodes: 
                                            self.add_parent(comp_node, parent)
                                    pim_parent_nodes = [] # reset pim parent nodes

                                    if "attn" in layers[layer_num].name:
                                        attn_remain = False
                                else: 
                                    if first_comp_node:
                                        if npu_group == 0:
                                            self.add_parent(comp_node, input_load_node)
                                        else:
                                            self.add_parent(comp_node, receive_input_node)
                                        if evict != None:
                                            self.add_parent(comp_node, evict)
                                        if load != None:
                                            self.add_parent(comp_node, load)
                                        if layers[layer_num].weight_memory_node != None:
                                            self.add_parent(comp_node, layers[layer_num].weight_memory_node)
                                        first_comp_node = False
                                    else:
                                        if layers[layer_num].weight_memory_node != None:
                                            self.add_parent(comp_node, layers[layer_num].weight_memory_node)
                                        if layers[layer_num - 1].comm_node != None:
                                            self.add_parent(comp_node, layers[layer_num - 1].comm_node)
                                        elif layers[layer_num - 1].comp_node != None:
                                            self.add_parent(comp_node, layers[layer_num - 1].comp_node)
                                        else:
                                            self.add_parent(comp_node, layers[layer_num - 2].comp_node)

                                # handle pim_compute_mode dependency & should not be remaining attention
                                if not attn_remain and len(pim_comp_nodes) != 0:
                                    if layers[layer_num].misc == "NONE": # no sub-batch interleaving
                                        for pim_comp in pim_comp_nodes:
                                            self.add_parent(comp_node, pim_comp)
                                        pim_comp_nodes = [] # reset pim comp nodes
                                    elif layers[layer_num].misc != last_batch_type:
                                        if len(past_pim_comp_nodes) != 0:
                                            for pim_comp in past_pim_comp_nodes:
                                                self.add_parent(comp_node, pim_comp)
                                        past_pim_comp_nodes = pim_comp_nodes # update past pim comp nodes
                                        pim_comp_nodes = [] # reset pim comp nodes
                                        last_batch_type = layers[layer_num].misc

                                encode_message(g, comp_node)

                            # PIM compute
                            if pim_start:
                                pim_comp_node = self.get_pim_compute_node(
                                    layers[layer_num].name,
                                    "PIM",
                                    layers[layer_num].comp_time,
                                    layers[layer_num].input_memory_loc,
                                    layers[layer_num].input_memory_size + layers[layer_num].output_memory_size # load/store from pim
                                )
                                pim_comp_nodes.append(pim_comp_node)
                                for parent in pim_parent_nodes:
                                    self.add_parent(pim_comp_node, parent)
                                encode_message(g, pim_comp_node)

                            # Communication (if required)
                            if layers[layer_num].comm_type != "NONE" and use_comm:
                                comm_coll_node = self.get_comm_coll_node(layers[layer_num].name, layers[layer_num].comm_type, layers[layer_num].comm_size, layers[layer_num].involved_dim)
                                # for j in range(self.num_dims):
                                # comm_coll_node.involved_dim.append(True)
                                layers[layer_num].comm_node = comm_coll_node
                                if layers[layer_num].comp_time != 0:
                                    self.add_parent(comm_coll_node, comp_node)
                                encode_message(g, comm_coll_node)
                            # add layer_num
                            layer_num += 1
                        # expert layer starts
                        elif layers[layer_num].is_expert:
                            # communication can happen even with one NPU in the group, for example, expert input gathering in data parallel
                            if expert_start == False and layers[layer_num].comm_size > 0 and layers[layer_num].comm_type != "NONE": 
                                # Start of expert, add ALLTOALL communication before expert computation
                                comm_coll_node = self.get_comm_coll_node("expert_start", layers[layer_num].comm_type, layers[layer_num].comm_size, layers[layer_num].involved_dim)
                                layers[layer_num].comm_node = comm_coll_node
                                self.add_parent(comm_coll_node, comp_node)
                                encode_message(g, comm_coll_node)
                            expert_start = True
                            # check expert end
                            if layers[layer_num].expert_num == 'END':
                                expert_start = False
                                layers[layer_num].comp_node = comp_node # is latest comp_node
                                # End of expert, add ALLTOALL communication after expert computation
                                if layers[layer_num].comm_size > 0 and layers[layer_num].comm_type != "NONE":
                                    comm_coll_node = self.get_comm_coll_node("expert_end", layers[layer_num].comm_type, layers[layer_num].comm_size, layers[layer_num].involved_dim)
                                    layers[layer_num].comm_node = comm_coll_node
                                    self.add_parent(comm_coll_node, comp_node)
                                    encode_message(g, comm_coll_node)
                                layer_num += 1
                                continue
                            # round robin assignment
                            expert_id = int(layers[layer_num].expert_num) % npus_per_group
                            if npu_offset != expert_id:
                                # go to next expert
                                while True:
                                    layer_num += 1
                                    if layers[layer_num].is_expert:
                                        break
                            else:
                                layers[layer_num].comp_node = comp_node # is latest comp_node
                                layer_num += 1
                        # pim layer starts
                        elif layers[layer_num].is_pim:
                            pim_start = True
                            # check attention end
                            if layers[layer_num].pim_num == 'END':
                                pim_start = False
                                if "attn" in layers[layer_num + 1].name:
                                    attn_remain = True # prefill attn remains
                                layer_num += 1
                                continue
                            # add pim parent nodes for dependency
                            elif int(layers[layer_num].pim_num) == 0:
                                if first_comp_node:
                                    if npu_group == 0:
                                        pim_parent_nodes.append(input_load_node)
                                    else:
                                        pim_parent_nodes.append(receive_input_node)
                                    if evict != None:
                                        pim_parent_nodes.append(evict)
                                    if load != None:
                                        pim_parent_nodes.append(load)
                                    # discarded weight parent because attention has no weight
                                    first_comp_node = False
                                else:
                                    # discarded weight parent because attention has no weight
                                    if layers[layer_num - 1].comm_node != None:
                                        pim_parent_nodes.append(layers[layer_num - 1].comm_node)
                                    elif layers[layer_num - 1].comp_node != None:
                                        pim_parent_nodes.append(layers[layer_num - 1].comp_node)
                                    else:
                                        pim_parent_nodes.append(layers[layer_num - 1].comp_node)
                            # round robin assignment
                            pim_id = int(layers[layer_num].pim_num) % npus_per_group
                            if npu_offset != pim_id:
                                # go to next attention
                                while True:
                                    layer_num += 1
                                    if layers[layer_num].is_pim:
                                        break
                            else:
                                layer_num += 1

                    # The frontend cuts stages on transformer-block boundaries,
                    # so the walk above lands exactly on the stage edge. Expert
                    # and PIM blocks live inside a block and can no longer be
                    # straddled; warn loudly if that ever stops holding.
                    if layer_num != layer_end:
                        print(f"Warning! pipeline stage {npu_group} walked to "
                              f"layer {layer_num}, expected {layer_end}")
                        layer_end = layer_num

                    if npu_group == (num_npu_group - 1):
                        # Store output (for the last layer)
                        # The last layer is the sampler: what crosses back to
                        # the host is its OUTPUT, the sampled token ids, 4 bytes
                        # per sequence.
                        #
                        # Do not switch this back to input_memory_size. That read
                        # was a deliberate down-scaling when lm_head was the last
                        # trace layer: its output is the logits
                        # (num_seqs * vocab_size * dtype) and billing those to CPU
                        # memory every iteration was a large overcharge, so the
                        # smaller input (the hidden state) stood in. Once sampler
                        # was appended as the new last layer the two swapped
                        # roles -- sampler's output is the token ids and its input
                        # is the logits -- so reading the input landed back on
                        # exactly the tensor the workaround existed to avoid.
                        output_store_node = self.get_memory_store_node(
                            layers[layer_end - 1].name,
                            "OUTPUT",
                            layers[layer_end - 1].output_memory_loc,
                            layers[layer_end - 1].output_memory_size,
                        )
                        # if pim_comp_nodes are not consumed yet, add dependency
                        if len(pim_comp_nodes) != 0:
                            for pim_comp in pim_comp_nodes:
                                self.add_parent(send_output_node, pim_comp)
                                pim_comp_nodes = []
                        if layers[layer_end - 1].comm_type != "NONE" and use_comm:
                            self.add_parent(output_store_node, comm_coll_node)
                        elif layers[layer_end - 1].comp_node != None:
                            self.add_parent(output_store_node, comp_node)
                        else:
                            self.add_parent(output_store_node, layers[layer_end - 2].comp_node)
                        encode_message(g, output_store_node)
                    else:
                        if layers[layer_end - 1].is_expert or layers[layer_end - 1].is_pim:
                            # Send output (to the next layer in another npu group)
                            send_output_node = self.get_comm_node(
                                is_send=True,
                                layer_name=layers[layer_end].name,
                                comm_type=layers[layer_end].comm_type,
                                comm_size=layers[layer_end].input_memory_size,
                                comm_src=npu_id,
                                comm_dst=npu_id + npus_per_group
                            )
                        else:
                            send_output_node = self.get_comm_node(
                                is_send=True,
                                layer_name=layers[layer_end - 1].name,
                                comm_type=layers[layer_end - 1].comm_type,
                                comm_size=layers[layer_end - 1].output_memory_size,
                                comm_src=npu_id,
                                comm_dst=npu_id + npus_per_group
                            )
                        # if pim_comp_nodes are not consumed yet, add dependency
                        if len(pim_comp_nodes) != 0:
                            for pim_comp in pim_comp_nodes:
                                self.add_parent(send_output_node, pim_comp)
                                pim_comp_nodes = []
                        if layers[layer_end - 1].comm_type != "NONE" and use_comm:
                            self.add_parent(send_output_node, comm_coll_node)
                        elif layers[layer_end - 1].comp_node != None:
                            self.add_parent(send_output_node, comp_node)
                        else:
                            self.add_parent(send_output_node, layers[layer_end - 2].comp_node)
                        encode_message(g, send_output_node)
    
    def convert_prefill(self, f: TextIOWrapper, num_layers: int, num_npu_group: int,
                        stage_boundaries: List[int] = None,
                        decode_npu_offset: int = None):
        layers: list[Layer] = self.get_layers(f)
        # There will be no pim operation in prefill (PIM cannot perform GEMM)

        # vllm: check eviction or load
        evict = None
        load = None
        ev_ld_cnt = 0
        for i in range(2):
            if 'kv_load' in layers[i].name:
                load = self.get_memory_load_node(
                            layers[i].name,
                            "WEIGHT",
                            layers[i].weight_memory_loc,
                            layers[i].weight_memory_size, # already per npu kv_cache size
                        )
            elif 'kv_evict' in layers[i].name:
                evict = self.get_memory_store_node(
                            layers[i].name,
                            "WEIGHT",
                            layers[i].weight_memory_loc,
                            layers[i].weight_memory_size,
                        )
            else:
                continue
            ev_ld_cnt += 1
            
        layers = layers[ev_ld_cnt:]
        num_layers -= ev_ld_cnt

        if self.num_npus % num_npu_group != 0: print("Warning! num_npus % num_npu_group != 0, Some npus won't do anything!")
        npus_per_group = self.num_npus // num_npu_group
        if npus_per_group == 1: # same as pipeline parallelism, ignore all reduce
            use_comm = False
        else:
            use_comm = True
        stage_edges = self.get_stage_edges(num_layers, num_npu_group,
                                           stage_boundaries or [])

        for npu_group in range(num_npu_group):
            for npu_offset in range(npus_per_group):
                # Re-read the authoritative edges per rank: the walk below may
                # rebind layer_end if it ever overruns.
                layer_start, layer_end = stage_edges[npu_group]
                npu_id = npu_group * npus_per_group + npu_offset + self.npu_offset
                output_filename1 = "%s.%d.et" % (self.output_filename, npu_id)
                decode_npu_id = (decode_npu_offset + npu_group * npus_per_group + npu_offset
                                 if decode_npu_offset is not None else npu_id + self.num_npus)
                output_filename2 = "%s.%d.et" % (self.output_filename, decode_npu_id)
                first_comp_node = True
                with open(output_filename1, "wb") as g, open(output_filename2, "wb") as s:
                    global_metadata = self.get_global_metadata()
                    encode_message(g, global_metadata)
                    encode_message(s, global_metadata)
                    if evict != None:
                        encode_message(g, evict)
                    if load != None:
                        encode_message(g, load)
                    if npu_group == 0:
                        # Load Input
                        input_load_node = self.get_memory_load_node(
                            layers[layer_start].name,
                            "INPUT",
                            layers[layer_start].input_memory_loc,
                            layers[layer_start].input_memory_size,
                        )
                        encode_message(g, input_load_node)                  
                    else:
                        if layers[layer_start].is_expert:
                            # Receive input (from the previous layer in another npu group)
                            receive_input_node = self.get_comm_node(
                                is_send=False,
                                layer_name=layers[layer_start-1].name,
                                comm_type=layers[layer_start-1].comm_type,
                                comm_size=layers[layer_start-1].output_memory_size,
                                comm_src=npu_id - npus_per_group,
                                comm_dst=npu_id
                            )
                            encode_message(g, receive_input_node)
                        else:
                            # Receive input (from the previous layer in another npu group)
                            receive_input_node = self.get_comm_node(
                                is_send=False,
                                layer_name=layers[layer_start].name,
                                comm_type=layers[layer_start].comm_type,
                                comm_size=layers[layer_start].input_memory_size,
                                comm_src=npu_id - npus_per_group,
                                comm_dst=npu_id
                            )
                            encode_message(g, receive_input_node)

                    expert_start = False
                    layer_num = layer_start
                    while expert_start or layer_num < layer_end:
                        if not layers[layer_num].is_expert:
                            if (self.local_offloading or layers[layer_num].weight_memory_loc != "LOCAL") and layers[layer_num].weight_memory_size > 0:
                                # Load weight (for weight offloading)
                                weight_load_node = self.get_memory_load_node(
                                    layers[layer_num].name,
                                    "WEIGHT",
                                    layers[layer_num].weight_memory_loc,
                                    layers[layer_num].weight_memory_size,
                                )
                                layers[layer_num].weight_memory_node = weight_load_node
                                if expert_start:
                                    self.add_parent(weight_load_node, comp_node) # dependent to previous comp_node due to gate function
                                encode_message(g, weight_load_node)
                            
                            # Compute
                            if layers[layer_num].comp_time != 0:
                                comp_node = self.get_comp_node(
                                    layers[layer_num].name, 
                                    layers[layer_num].comp_time)
                                layers[layer_num].comp_node = comp_node

                                if first_comp_node:
                                    if npu_group == 0:
                                        self.add_parent(comp_node, input_load_node)
                                    else:
                                        self.add_parent(comp_node, receive_input_node)
                                    if evict != None:
                                        self.add_parent(comp_node, evict)
                                    if load != None:
                                        self.add_parent(comp_node, load)
                                    if layers[layer_num].weight_memory_node != None:
                                        self.add_parent(comp_node, layers[layer_num].weight_memory_node)
                                    first_comp_node = False
                                else:
                                    if layers[layer_num].weight_memory_node != None:
                                        self.add_parent(comp_node, layers[layer_num].weight_memory_node)
                                    if layers[layer_num - 1].comm_node != None:
                                        self.add_parent(comp_node, layers[layer_num - 1].comm_node)
                                    elif layers[layer_num - 1].comp_node != None:
                                        self.add_parent(comp_node, layers[layer_num - 1].comp_node)
                                    else:
                                        self.add_parent(comp_node, layers[layer_num - 2].comp_node)
                                
                                encode_message(g, comp_node)

                                # Send KV cache after each kv_proj.
                                # comm_size comes from the trace's comm_size
                                # column, which the frontend fills with the
                                # per-layer, per-rank K+V bytes. It used to read
                                # output_memory_size, i.e. the whole QKV
                                # activation, which shipped Q as well and
                                # overstated the transfer by
                                # (q_dim + 2*kv_dim) / (2*kv_dim) -- 3x for
                                # Llama-3.1-8B -- and ignored kv_cache_dtype.
                                if "v_proj" in layers[layer_num].name and layers[layer_num].comm_size > 0:
                                    send_kv_node = self.get_comm_node(
                                        is_send=True,
                                        layer_name="kv_proj",
                                        comm_type=layers[layer_num].comm_type,
                                        comm_size=layers[layer_num].comm_size,
                                        comm_src=npu_id,
                                        comm_dst=decode_npu_id,
                                        id=layer_num
                                    )
                                    self.add_parent(send_kv_node, comp_node)
                                    encode_message(g, send_kv_node)

                                    recv_kv_node = self.get_comm_node(
                                        is_send=False,
                                        layer_name="kv_proj",
                                        comm_type=layers[layer_num].comm_type,
                                        comm_size=layers[layer_num].comm_size,
                                        comm_src=npu_id,
                                        comm_dst=decode_npu_id,
                                        id=layer_num
                                    )
                                    encode_message(s, recv_kv_node)

                            # Communication (if required)
                            if layers[layer_num].comm_type != "NONE" and use_comm:
                                comm_coll_node = self.get_comm_coll_node(layers[layer_num].name, layers[layer_num].comm_type, layers[layer_num].comm_size, layers[layer_num].involved_dim)
                                # for j in range(self.num_dims):
                                # comm_coll_node.involved_dim.append(True)
                                layers[layer_num].comm_node = comm_coll_node
                                if layers[layer_num].comp_time != 0:
                                    self.add_parent(comm_coll_node, comp_node)
                                encode_message(g, comm_coll_node)
                            # add layer_num
                            layer_num += 1
                        # expert layer starts
                        elif layers[layer_num].is_expert:
                            # communication can happen even with one NPU in the group, for example, expert input gathering in data parallel
                            if expert_start == False and layers[layer_num].comm_size > 0 and layers[layer_num].comm_type != "NONE": 
                                # Start of expert, add ALLTOALL communication before expert computation
                                comm_coll_node = self.get_comm_coll_node("expert_start", layers[layer_num].comm_type, layers[layer_num].comm_size, layers[layer_num].involved_dim)
                                layers[layer_num].comm_node = comm_coll_node
                                self.add_parent(comm_coll_node, comp_node)
                                encode_message(g, comm_coll_node)
                            expert_start = True
                            # check expert end
                            if layers[layer_num].expert_num == 'END':
                                expert_start = False
                                layers[layer_num].comp_node = comp_node # is latest comp_node
                                # End of expert, add ALLTOALL communication after expert computation
                                if layers[layer_num].comm_size > 0 and layers[layer_num].comm_type != "NONE":
                                    comm_coll_node = self.get_comm_coll_node("expert_end", layers[layer_num].comm_type, layers[layer_num].comm_size, layers[layer_num].involved_dim)
                                    layers[layer_num].comm_node = comm_coll_node
                                    self.add_parent(comm_coll_node, comp_node)
                                    encode_message(g, comm_coll_node)
                                layer_num += 1
                                continue
                            # round robin assignment
                            expert_id = int(layers[layer_num].expert_num) % npus_per_group
                            if npu_offset != expert_id:
                                # go to next expert
                                while True:
                                    layer_num += 1
                                    if layers[layer_num].is_expert:
                                        break
                            else:
                                layers[layer_num].comp_node = comp_node # is latest comp_node
                                layer_num += 1

                    # The frontend cuts stages on transformer-block boundaries,
                    # so the walk above lands exactly on the stage edge. Expert
                    # and PIM blocks live inside a block and can no longer be
                    # straddled; warn loudly if that ever stops holding.
                    if layer_num != layer_end:
                        print(f"Warning! pipeline stage {npu_group} walked to "
                              f"layer {layer_num}, expected {layer_end}")
                        layer_end = layer_num

                    if npu_group == (num_npu_group - 1):
                        # Send output (for the last layer, to the paired decode npu)
                        send_output_node = self.get_comm_node(
                            is_send=True,
                            layer_name=layers[layer_end - 1].name,
                            comm_type=layers[layer_end - 1].comm_type,
                            comm_size=layers[layer_end - 1].output_memory_size,
                            comm_src=npu_id,
                            comm_dst=npu_id + self.num_npus
                        )
                        if layers[layer_end - 1].comm_type != "NONE" and use_comm:
                            self.add_parent(send_output_node, comm_coll_node)
                        elif layers[layer_end - 1].comp_node != None:
                            self.add_parent(send_output_node, comp_node)
                        else:
                            self.add_parent(send_output_node, layers[layer_end - 2].comp_node)
                        encode_message(g, send_output_node)
                        # paired decode npu receive output
                        recv_output_node = self.get_comm_node(
                            is_send=False,
                            layer_name=layers[layer_end - 1].name,
                            comm_type=layers[layer_end - 1].comm_type,
                            comm_size=layers[layer_end - 1].output_memory_size,
                            comm_src=npu_id,
                            comm_dst=npu_id + self.num_npus
                        )
                        encode_message(s, recv_output_node)
                    else:
                        if layers[layer_end - 1].is_expert:
                            # Send output (to the next layer in another npu group)
                            send_output_node = self.get_comm_node(
                                is_send=True,
                                layer_name=layers[layer_end].name,
                                comm_type=layers[layer_end].comm_type,
                                comm_size=layers[layer_end].input_memory_size,
                                comm_src=npu_id,
                                comm_dst=npu_id + npus_per_group
                            )
                        else:
                            send_output_node = self.get_comm_node(
                                is_send=True,
                                layer_name=layers[layer_end - 1].name,
                                comm_type=layers[layer_end - 1].comm_type,
                                comm_size=layers[layer_end - 1].output_memory_size,
                                comm_src=npu_id,
                                comm_dst=npu_id + npus_per_group
                            )
                        if layers[layer_end - 1].comm_type != "NONE" and use_comm:
                            self.add_parent(send_output_node, comm_coll_node)
                        elif layers[layer_end - 1].comp_node != None:
                            self.add_parent(send_output_node, comp_node)
                        else:
                            self.add_parent(send_output_node, layers[layer_end - 2].comp_node)
                        encode_message(g, send_output_node)

    def convert_event(self, f: TextIOWrapper, num_layers: int):
        layers: list[Layer] = self.get_layers(f)
        for npu_id in range(self.num_npus):
            output_filename = "%s.%d.et" % (self.output_filename, npu_id)
            with open(output_filename, "wb") as g:
                global_metadata = self.get_global_metadata()
                encode_message(g, global_metadata)
                for idx, layer in enumerate(layers):
                    comp_node = self.get_comp_node(
                    layer.name, 
                    layer.comp_time)
                    layer.comp_node = comp_node
                    encode_message(g, comp_node)

    def convert(self):
        with open(self.input_filename, "r") as f:
            first_line = f.readline().strip().split()
            execution_type = first_line[0]

            # The first line carries "key: value" pairs after the execution
            # type, e.g. "model_parallel_NPU_group: 4  pp_stage_boundaries: 73,145,217".
            header = {}
            fields = first_line[1:]
            for i in range(0, len(fields) - 1, 2):
                if fields[i].endswith(":"):
                    header[fields[i][:-1]] = fields[i + 1]

            num_npu_group = int(header.get("model_parallel_NPU_group", 0))
            boundary_str = header.get("pp_stage_boundaries", "")
            stage_boundaries = (
                [int(b) for b in boundary_str.split(",")] if boundary_str else []
            )

            second_line = f.readline().strip()
            num_layers = int(second_line)

            third_line = f.readline() # This is for the table header, so just ignore it

            if execution_type == "COLOCATED":
                if num_npu_group <= 0:
                    raise ValueError(f"model_parallel_NPU_group <= 0")
                self.convert_common(f, num_layers, num_npu_group, stage_boundaries)
            elif execution_type == "PREFILL":
                if num_npu_group <= 0:
                    raise ValueError(f"model_parallel_NPU_group <= 0")
                target = int(header["pd_decode_npu_offset"]) if "pd_decode_npu_offset" in header else None
                self.convert_prefill(f, num_layers, num_npu_group, stage_boundaries, target)
            elif execution_type == "DECODE":
                if num_npu_group <= 0:
                    raise ValueError(f"model_parallel_NPU_group <= 0")
                self.convert_common(f, num_layers, num_npu_group, stage_boundaries)
            elif execution_type == "EVENT":
                self.convert_event(f, num_layers)
            else:
                raise ValueError(f"Unsupported execution type, {execution_type}")

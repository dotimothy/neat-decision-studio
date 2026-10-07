# Building the CLM runtime off the board

`runtime/` builds on the Modalix DevKit against the LLiMa runtime library. These stand-ins let
the same sources be built and run on any machine, to try the part of the runtime that is not
the accelerator: the layout of a question, which texts share a pass, the attention mask, where
each text's hidden state is read, the heads and the memory of what has been embedded.

    python3 tools/clm_mock/make_model.py /tmp/clm-mock-model      # needs numpy
    g++ -std=c++20 -O1 -DFMT_HEADER_ONLY -I tools/clm_mock -I runtime/include \
        runtime/src/*.cpp -o /tmp/laya-mock                        # needs fmt and nlohmann-json headers
    /tmp/laya-mock run /tmp/clm-mock-model --state "..." --question '{"type": "noul", "instructions": "..."}'

The stand-in graph is not Qwen3 and its answers mean nothing. What it keeps is what packing
depends on: a position's output is made only of the positions its mask row opens. So
`laya hidden --packed 1` and `--packed 0` give the same numbers there exactly when the mask
and the buffers are handled right, which is the check.

# Demo

End-to-end demo: train a tiny MLP, export to `.cinf`, run inference in C.

## Run via Make

```
make demo
```

This runs `demos/mlp_demo.py` to export `build/mlp.cinf` and then
`build/run_inference` to load and run it.

## Run manually

```
python demos/mlp_demo.py
cc -std=c11 -Wall -O2 -Iinclude demos/run_inference.c -Lbuild -lcinfer -lm -o build/run_inference
./build/run_inference build/mlp.cinf
```

The MLP architecture is `Linear(8 -> 6) -> ReLU -> Linear(6 -> 4) ->
Softmax`, matching the end-to-end parity test fixture.

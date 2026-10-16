"""Entry point for a Hugging Face *Gradio-SDK* Space (Docker SDK is paid, CPU hardware is unavailable
on this free account, and ZeroGPU Spaces only stay up if Gradio is launched with an @spaces.GPU function).

So: launch a minimal Gradio Blocks (with a no-op @spaces.GPU function to satisfy ZeroGPU), then put the
real TumorVision FastAPI routes in front of Gradio's own routes. The models run on CPU as before."""
import gradio as gr

try:
    import spaces

    @spaces.GPU
    def _zerogpu_placeholder():
        return "ok"
except ImportError:   # local run / CPU hardware without the `spaces` package
    def _zerogpu_placeholder():
        return "ok"

import backend.app as tv

with gr.Blocks(title="TumorVision (Gradio shell)") as demo:
    gr.Markdown("TumorVision API shell - the real app is served at `/`.")
    out = gr.Textbox(visible=False)
    gr.Button("ping", visible=False).click(_zerogpu_placeholder, None, out)

tv.load_models()   # normally done by FastAPI's startup event, which does not fire on Gradio's app
demo.queue()
demo.launch(server_name="0.0.0.0", server_port=7860, prevent_thread_lock=True, ssr_mode=False)  # SSR (Node) would swallow our routes

skip = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
ours = [r for r in tv.app.routes if getattr(r, "path", None) not in skip]
demo.app.router.routes[0:0] = ours      # our routes win over Gradio's catch-alls
demo.block_thread()

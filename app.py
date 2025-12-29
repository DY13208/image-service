from config import GRADIO_SERVER_NAME, GRADIO_SERVER_PORT
from ui import build_demo


def main():
    demo = build_demo()
    demo.launch(
        server_name=GRADIO_SERVER_NAME,
        server_port=GRADIO_SERVER_PORT,
        show_error=True,
        debug=True,
    )


if __name__ == "__main__":
    main()

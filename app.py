from run.run import create_app

app = create_app()

if __name__ == "__main__":
    from run.run import start_app
    start_app(app)
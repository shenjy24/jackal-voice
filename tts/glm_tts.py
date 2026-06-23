from gradio_client import Client

client = Client("https://gswyhq-glm-tts.ms.show/")
result = client.predict(
	text="哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
	instruction="Hello!!",
	language_zh="中文",
	preset_name="Hello!!",
	api_name="/generate_voice_design_fn"
)
print(result)
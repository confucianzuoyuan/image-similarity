# 使用 Flash 实现 Web 服务器

from waitress import serve
import web_app

serve(web_app.app, port=9001, threads=6)

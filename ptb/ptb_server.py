from http.server import HTTPServer,BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import threading
import urllib

class Handler(BaseHTTPRequestHandler):
	callbackfn = None
	def do_GET(self):
		self.send_response(200)
		self.end_headers()
		if not self.path.startswith('/query/'):
			indexf = open("ptb/index.html")
			self.wfile.write(index.read())
			indexf.close()
			return
		else:
			sentence = urllib.unquote(self.path.replace("/query/",""))
			print("sentence is"+self.path)
			completion = sentence
			completion = self.callbackfn(sentence)
			self.wfile.write(completion)
			self.wfile.write('\n')
			return

class ThreadHTTPServer(ThreadingMixIn,HTTPServer):
	"""Handle requests in a seperate thread."""
	def start_server(fn = None):
		Handler.callbackfn = fn
		server = HTTPServer(('',8080),Handler)
		#server.handler.callbackfn = fn
		print('Starting server with callback fn,use <Ctrl-C> to stop',fn)
		server.serve_forever()

if __name__=='__main__':
	start_server()
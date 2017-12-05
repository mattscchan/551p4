import argparse
from email.parser import Parser
import sys
import re
import HTMLParser

reload(sys)
sys.setdefaultencoding("utf8")

def remove_html(msg):
	style_re = re.compile(r'(<style>.*?</style>)',flags=re.DOTALL)
	msg = style_re.sub("", msg)
	tag_re = re.compile(r'(<!--.*?-->|<[^>]*>)')
	msg = tag_re.sub("", msg)
	return HTMLParser.HTMLParser().unescape(msg)

def flatten(msg):
	if msg.is_multipart():
		mlist =  map((lambda m: flatten(m)), msg.get_payload())
		return [item for sublist in mlist for item in sublist]
	else:
		charset = msg.get_param("charset")
		# print "Charset: " + str(charset)
		payload = msg.get_payload()
		if charset != None:
		  try:
		    payload = payload.decode(charset, "replace")
		  except LookupError:
		    payload = msg.get_payload().decode("utf8", "replace")
		else:
		  payload = payload.decode("utf8", "replace") 

		# initially checked for "text/html"
		# but removed check because some e-mails are badly formatted
		# so just stripping everything of html
		return [remove_html(payload)]

def process(f):
	msg = Parser().parse(f)

	subject_line = msg.get("Subject", "").decode("utf8", "replace")

	contents = flatten(msg)
	new_lines = [subject_line+"\n"] + contents 

	full_message = "".join(new_lines)
	return full_message.encode("utf-8")

def main(args):

	for index in range(1, 75420):
		with open(args.filename + str(index), 'r') as f:
			with open('./clean/mail_'+str(index)+'.txt', 'w') as f2:
				f2.write(process(f))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('filename')
	args = parser.parse_args()
	main(args)
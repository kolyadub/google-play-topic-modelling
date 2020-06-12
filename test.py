# coding=utf8
import json

filename = 'file.txt'

commands = ''
with open(filename) as fh:
    for line in fh:
        commands += line

dic = json.loads(commands)
print(dic['app_title'])
# print(type(json.loads(json.dumps(commands, sort_keys=False))))
# print(str(json.loads((json.dumps(commands, indent=2, sort_keys=False)))).replace("\\",""))

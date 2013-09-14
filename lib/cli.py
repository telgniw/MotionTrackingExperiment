#!/usr/bin/env python
import cmd, threading

class Cmd(cmd.Cmd):
    prompt  = '/_> '

    def default(self, line):
        args = self.parseline(line)
        if args[0] == '\\':
            pass
        else:
            print 'Error: unknown syntax "%s"' % line

    def do_exit(self, line):
        return True

    def do_camera(self, cid):
        try:
            cid = int(cid)
        except:
            cid = None

        self.main.set_camera_id(cid)

class CmdRunner(threading.Thread):
    def __init__(self, main):
        super(CmdRunner, self).__init__()
        self.cmd = Cmd()
        self.cmd.main = main

    def run(self):
        self.cmd.cmdloop()

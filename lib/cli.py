#!/usr/bin/env python
import cmd, threading
import os
from datetime import datetime

class Cmd(cmd.Cmd):
    prompt  = '/_> '

    def default(self, line):
        args = self.parseline(line)
        if args[0] == '\\':
            pass
        else:
            print 'Error: unknown syntax "%s"' % line

    def do_exit(self, _):
        return True

    def do_add(self, filename):
        if os.path.exists(filename):
            filename = os.path.abspath(filename)
            self.main.do_add_target(filename)
        else:
            print 'Error: file not found "%s"' % filename

    def do_camera(self, cid):
        try:
            cid = int(cid)
        except:
            cid = None

        self.main.do_switch_camera(cid)
        print 'Switching to camera', cid

    def do_snapshot(self, filename):
        if filename:
            filename = os.path.abspath(filename)
        else:
            filename = datetime.now().strftime('frame_%Y%m%d_%H%M%S.png')

        self.main.do_snapshot(filename)
        print 'Saving to', filename

class CmdRunner(threading.Thread):
    def __init__(self, main):
        super(CmdRunner, self).__init__()
        self.cmd = Cmd()
        self.cmd.main = main

    def run(self):
        self.cmd.cmdloop()

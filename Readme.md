Motion Tracking Experiment
==========================

Blah blah blah.

Method Implemented
------------------

Reference:

- http://derek.simkowiak.net/motion-tracking-with-python/
- http://code.google.com/p/find-object/

Video Format
------------

Convert "mov" file to "avi" file.

    # Install mplayer, which includes mencoder.
    brew install mplayer

    # Convert to "avi" file without audio channel.
    mencoder <input.mov> -ovc raw -nosound -vf format=i420 -o <output.avi>

Reference: http://opencv.willowgarage.com/wiki/VideoCodecs

License
-------

2013 [CC by-NC-SA] | [Yi Huang]

[CC by-NC-SA]: http://creativecommons.org/licenses/by-nc-sa/3.0/tw/
[Yi Huang]: http://github.com/telgniw

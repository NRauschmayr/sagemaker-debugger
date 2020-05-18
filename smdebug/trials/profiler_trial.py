import glob
from os.path import join

from smdebug.core.reader import FileReader
from smdebug.core.writer import FileWriter
from smdebug.core.logger import get_logger

class ProfilerTrial():
    def __init__(self, path, dirname=None):
        self.logger = get_logger()
        self.event_files = glob.glob(join(path, "events", "*", "*.tfevents"), recursive=True)
        self.event_files += glob.glob(join(path, "events", "*", "*.tfevents.tmp"), recursive=True)
        self.events = self.read_scalars()
        
    def read_scalars(self, regex_list=None):
        events = list()
        
        for f in self.event_files:
            fr = FileReader(f)
            events += fr.read_events(regex_list=regex_list)
        # Create a dict of scalar events.
        scalar_events = dict()
        if len(events) > 0:
          t0 = events[0]["timestamp"]
        for index, x in enumerate(events):
            event_name = str(x["name"])
            if event_name not in scalar_events:
                scalar_events[event_name] = list()
            scalar_events[event_name].append((x["timestamp"] - t0, x["value"][0]))
        
        return scalar_events

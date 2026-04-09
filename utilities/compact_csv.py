import pickle
import os


class PickleRowWriter:
    def __init__(self, file_obj):
        # Expects a file opened in 'ab' or 'wb' mode
        self.file = file_obj
        
    def writerow(self, row):
        pickle.dump(row, self.file)


class PickleRowReader:
    def __init__(self, file_obj):
        # Expects a file opened in 'rb' mode
        self.file = file_obj

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return pickle.load(self.file)
        except EOFError:
            # Reached end of file. 
            # We raise StopIteration, but we DO NOT close the file here.
            # The external user is responsible for closing it.
            raise StopIteration
        

def writer(file_obj):
    return PickleRowWriter(file_obj)


def reader(file_obj):
    return PickleRowReader(file_obj)



class Episode_Recorder:

    def __init__(self, record_statistic_dir, headers):
        self.record_statistic_dir = record_statistic_dir
        if record_statistic_dir is not None:
            os.makedirs(record_statistic_dir, exist_ok=True)

        stat_file_name = os.path.join(self.record_statistic_dir, f"episode_statistics.csv")
        if not os.path.exists(stat_file_name):
            with open(stat_file_name, mode='wb') as stat_file:
                w = writer(stat_file)
                w.writerow(headers)

        self.last_row_values = ["" for _ in headers] # List to store the last row values, initialized to None
        self.rows = []
    

    def record(self, row):
        if self.record_statistic_dir is None:
            return
        # Update last row values with the new row values, then append a copy of the last row values to rows
        for i in range(len(row)):
            if row[i] is not None:
                self.last_row_values[i] = row[i]
        # Append a copy of the last row values to rows. We use copy to ensure that we are appending the current state of last_row_values, not a reference to it.
        self.rows.append(self.last_row_values.copy())

        
    def write(self):
        if self.record_statistic_dir is None:
            return
        
        stat_file_name = os.path.join(self.record_statistic_dir, f"episode_statistics.csv")
        with open(stat_file_name, mode='ab') as stat_file:
            w = writer(stat_file)
            for row in self.rows:
                w.writerow(row)
        self.rows = []
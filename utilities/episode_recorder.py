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


# header spec: the column records increments and the file stores their running sum
CUMSUM = "cumsum"


def prefixed_headers(prefixes, columns):
    """Headers "{prefix}/{name}" for every prefix (e.g. game id) and column; a column is a name or a (name, spec) tuple."""
    columns = [c if isinstance(c, tuple) else (c, None) for c in columns]
    return [(f"{prefix}/{name}", spec) for prefix in prefixes for name, spec in columns]



class Episode_Recorder:

    def __init__(self, record_statistic_dir, headers):
        """headers: column names, or (name, spec) tuples. spec "cumsum" records the running sum of the column,
        continued from the last value in an existing file, so counters like env_steps survive a resumed run."""
        self.record_statistic_dir = record_statistic_dir
        if record_statistic_dir is not None:
            os.makedirs(record_statistic_dir, exist_ok=True)

        self.headers = [h[0] if isinstance(h, tuple) else h for h in headers]
        self.specs = [h[1] if isinstance(h, tuple) else None for h in headers]
        self.cumsums = [0 if spec == "cumsum" else None for spec in self.specs]

        stat_file_name = os.path.join(self.record_statistic_dir, f"episode_statistics.csv")
        if not os.path.exists(stat_file_name):
            with open(stat_file_name, mode='wb') as stat_file:
                w = writer(stat_file)
                w.writerow(self.headers)
        elif any(spec == "cumsum" for spec in self.specs):
            with open(stat_file_name, mode='rb') as stat_file:
                r = reader(stat_file)
                next(r, None)
                for row in r:
                    for i, value in enumerate(row[:len(self.cumsums)]):
                        if self.cumsums[i] is not None and value != "":
                            self.cumsums[i] = value

        self.rows = []
    

    def record(self, row):
        if self.record_statistic_dir is None:
            return
        # Update last row values with the new row values, then append a copy of the last row values to rows
        last_row_values = []
        for i in range(len(row)):
            if row[i] is not None:
                if i < len(self.cumsums) and self.cumsums[i] is not None:
                    self.cumsums[i] += row[i]
                    last_row_values.append(self.cumsums[i])
                else:
                    last_row_values.append(row[i])
            else:
                last_row_values.append("")
        self.rows.append(last_row_values)

        
    def write(self):
        if self.record_statistic_dir is None:
            return
        
        stat_file_name = os.path.join(self.record_statistic_dir, f"episode_statistics.csv")
        with open(stat_file_name, mode='ab') as stat_file:
            w = writer(stat_file)
            for row in self.rows:
                w.writerow(row)
        self.rows = []
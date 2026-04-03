import pickle


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
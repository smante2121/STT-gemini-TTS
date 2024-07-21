# Description: This file contains the MicrophoneStream class which is used to stream audio data from the microphone.
import queue
import sounddevice as sd

class MicrophoneStream(object): # Create a MicrophoneStream class
    def __init__(self, rate, chunk): # Initialize the MicrophoneStream class
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self._closed = True

    def __enter__(self): # Enter the context
        self._closed = False
        return self

    def __exit__(self, type, value, traceback): # Exit the context
        self._closed = True

    def start(self): # Start the stream
        print("Stream started")
        self._closed = False

    def stop(self): # Stop the stream
        print("Stream has been closed")
        self._closed = True

    def generator(self): # Generate audio chunks from the mic
        # generate audio chunks from the mic
        with sd.InputStream(samplerate=self._rate, channels=1, dtype='int16', blocksize=self._chunk) as stream:
            while not self._closed: # while the stream is open
                data, overflowed = stream.read(self._chunk) # read a chuck of audio
                if overflowed: # if the buffer overflowed
                    print("Audio buffer overflowed") # notify if buffer overflow
                if data is not None and len(data) > 0: # if there is audio data
                    self._buff.put(data.tobytes()) # put data into queue
                    yield self._buff.get() # yield audio data from the queue
                else:
                    print("No audio data") # notify if no audio data
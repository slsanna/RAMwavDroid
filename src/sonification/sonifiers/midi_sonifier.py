import mido
import logging
from .base_sonify import BaseSonifier

logger = logging.getLogger(__name__)

class MidiSonifier(BaseSonifier):

    def __init__(self, sample_rate=44100):
        super().__init__(sample_rate)
        self.ticks_per_beat = 480
        self.tempo = 500000

    def byteToNote(self, byte_val):
        return min(byte_val // 2, 127)

    def byteToVelocity(self, byte_val):
        return 40 + (byte_val * 60) // 255

    def getFileExtension(self):
        return ".mid"

    def sonify(self, binary_data, output_path):
        try:
            mid = mido.MidiFile(ticks_per_beat=self.ticks_per_beat)
            track = mido.MidiTrack()
            mid.tracks.append(track)
            
            track.append(mido.MetaMessage('set_tempo', tempo=self.tempo))
            
            ticks_per_note = 5
            instrument_change_interval = 256
            
            for i in range(0, len(binary_data) - 1, 2):
                if i > 0 and i % instrument_change_interval == 0 and i+1 < len(binary_data):
                    program = binary_data[i] % 127
                    track.append(mido.Message('program_change', program=program, time=0))
                
                if i+1 < len(binary_data):
                    note = self.byteToNote(binary_data[i])
                    velocity = self.byteToVelocity(binary_data[i + 1])
                    
                    track.append(mido.Message('note_on', note=note, velocity=velocity, time=0))
                    track.append(mido.Message('note_off', note=note, velocity=0, time=ticks_per_note))
            
            mid.save(output_path)
            logger.info(f"MIDI file created: {output_path} ({len(binary_data)} bytes processed)")
            return True
            
        except Exception as e:
            logger.error(f"Error creating MIDI file: {str(e)}")
            return False


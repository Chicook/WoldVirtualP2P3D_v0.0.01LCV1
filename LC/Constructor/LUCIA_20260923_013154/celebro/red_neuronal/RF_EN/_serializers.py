from __future__ import annotations
import pickle
class SerializationHelper:
    @staticmethod
    def serialize_neuron(neurona) -> bytes:
        state = {'input_size': neurona.input_size, 'output_size': neurona.output_size, 'nombre': neurona.nombre, 'pasos': neurona.pasos}
        if neurona.pesos is not None: state['pesos'] = neurona.pesos.tobytes()
        if neurona.sesgo is not None: state['sesgo'] = neurona.sesgo.tobytes()
        return pickle.dumps(state)
    @staticmethod
    def deserialize_neuron(data: bytes, neurona_class):
        state = pickle.loads(data); return neurona_class(state['input_size'], state['output_size'])
    @staticmethod
    def save_to_file(neurona, filepath: str) -> None:
        with open(filepath, 'wb') as f: f.write(SerializationHelper.serialize_neuron(neurona))
    @staticmethod
    def load_from_file(filepath: str, neurona_class):
        with open(filepath, 'rb') as f: return SerializationHelper.deserialize_neuron(f.read(), neurona_class)

logger = logging.getLogger('RFENRN1')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.info("Paquete RFENRN1 inicializado para LucIA Reinforcement Learning")


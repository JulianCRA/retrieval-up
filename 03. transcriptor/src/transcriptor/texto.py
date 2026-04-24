INFO_ASR = """
INFO ASR
- Vosk: Un modelo de ASR de código abierto que ofrece transcripción en tiempo real y soporte para múltiples idiomas.
- Wav2Vec: Un modelo de ASR desarrollado por Facebook AI que utiliza aprendizaje profundo para transcribir audio a texto con alta precisión.
- Cohere: Un modelo de ASR basado en inteligencia artificial que ofrece transcripción de alta calidad y soporte para varios idiomas.
- Whisper: Un modelo de ASR desarrollado por OpenAI que utiliza aprendizaje profundo para transcribir audio a texto con alta precisión, especialmente en entornos ruidosos.
- Qwen: Un modelo de ASR desarrollado por Alibaba que ofrece transcripción de alta calidad y soporte para varios idiomas, con un enfoque en la eficiencia y la escalabilidad.
"""

INFO_VOSK = """
VOSK
Motor de ASR de código abierto con modelos ligeros optimizados para ejecución local en tiempo real. Soporta más de 20 idiomas y permite transcripción offline sin conexión a internet. Los modelos varían en tamaño (40 MB – 1,8 GB) según la precisión deseada.

Recursos mínimos:
  - CPU: x86-64 de doble núcleo (sin GPU necesaria)
  - RAM: 1 GB
  - Disco: 150 MB (modelo "small" en español)
  - GPU: no requerida

Recursos recomendados:
  - CPU: x86-64 de cuatro núcleos o más
  - RAM: 4 GB
  - Disco: 2 GB (modelo completo para mayor precisión)
  - GPU: no requerida (Vosk no aprovecha GPU)
"""

INFO_WAC2VEC = """
WAV2VEC 2.0 — jonatasgrosman/wav2vec2-large-xlsr-53-spanish
Modelo de ASR basado en aprendizaje auto-supervisado, afinado por jonatasgrosman sobre el checkpoint XLSR-53 de Facebook AI para español. Disponible en Hugging Face (jonatasgrosman/wav2vec2-large-xlsr-53-spanish). Ofrece alta precisión en español con inferencia local sin necesidad de conexión a internet.

Recursos mínimos:
  - CPU: x86-64 de cuatro núcleos
  - RAM: 8 GB
  - Disco: 1,3 GB (pesos del modelo large-xlsr)
  - GPU: no estrictamente requerida (inferencia en CPU posible pero lenta)

Recursos recomendados:
  - CPU: octa-núcleo o superior
  - RAM: 16 GB
  - Disco: 2 GB
  - GPU: NVIDIA con 6 GB de VRAM y CUDA 11+ (acelera la inferencia ×5–×10 respecto a CPU)
"""

INFO_COHERE = """
COHERE (API de transcripción)
Servicio de ASR basado en la nube de Cohere. La transcripción se realiza en los servidores de Cohere mediante llamadas a su API REST; el cliente local solo envía el audio y recibe el texto. Requiere conexión a internet y una API key activa.

Recursos mínimos (lado cliente):
  - CPU: cualquier procesador capaz de ejecutar Python 3.8+
  - RAM: 512 MB
  - Disco: < 50 MB (solo la librería del cliente)
  - GPU: no requerida
  - Red: conexión estable a internet (latencia < 200 ms recomendada)

Recursos recomendados (lado cliente):
  - CPU: dual-core moderno
  - RAM: 2 GB
  - Red: banda ancha ≥ 10 Mbps para archivos de audio largos
  - Nota: el rendimiento real depende de la capacidad del servidor de Cohere y del plan contratado.
"""

INFO_WHISPER = """
WHISPER (OpenAI)
Modelo de ASR de código abierto entrenado con 680 000 horas de audio multilingüe. Ofrece transcripción y traducción automática con alta tolerancia al ruido. Los modelos disponibles en esta aplicación son: tiny, base, small y large-v3-turbo.

  tiny
    Uso: pruebas rápidas y dispositivos muy limitados.
    Mínimos : CPU dual-core, RAM 1 GB, Disco 150 MB, GPU no requerida.
    Recomendado: CPU quad-core, RAM 4 GB, GPU 2 GB VRAM.

  base
    Uso: balance entre velocidad y precisión en hardware modesto.
    Mínimos : CPU quad-core, RAM 2 GB, Disco 290 MB, GPU no requerida.
    Recomendado: CPU quad-core, RAM 4 GB, GPU 2 GB VRAM.

  small
    Uso: buena precisión con recursos moderados.
    Mínimos : CPU quad-core, RAM 4 GB, Disco 970 MB, GPU 2 GB VRAM (o CPU lento).
    Recomendado: CPU octa-core, RAM 8 GB, GPU 4 GB VRAM y CUDA 11.8+.

  large-v3-turbo
    Uso: máxima precisión multilingüe con velocidad optimizada respecto a large-v3.
    Mínimos : CPU octa-core, RAM 8 GB, Disco 1,6 GB, GPU 6 GB VRAM y CUDA 11.8+.
    Recomendado: CPU 12 núcleos, RAM 16 GB, GPU NVIDIA RTX 3080/4070 (10–12 GB VRAM).
    Nota: sin GPU, la inferencia puede tardar varias veces la duración del audio.
"""

INFO_QWEN = """
QWEN3-ASR (Alibaba)
Modelos ASR de la familia Qwen3 especializados en transcripción multilingüe de alta precisión. Basados en arquitectura transformer ligera optimizada para velocidad y bajo consumo de recursos. Los modelos disponibles en esta aplicación son: Qwen3-ASR-0.6B y Qwen3-ASR-1.7B.

  Qwen3-ASR-0.6B
    Uso: transcripción eficiente en hardware modesto o con recursos limitados.
    Mínimos : CPU quad-core, RAM 4 GB, Disco 1,2 GB, GPU no requerida.
    Recomendado: CPU octa-core, RAM 8 GB, GPU NVIDIA 4 GB VRAM y CUDA 11.8+.

  Qwen3-ASR-1.7B
    Uso: mayor precisión y mejor manejo de vocabulario complejo y acentos.
    Mínimos : CPU octa-core, RAM 8 GB, Disco 3,5 GB, GPU NVIDIA 6 GB VRAM y CUDA 11.8+.
    Recomendado: CPU 12 núcleos, RAM 16 GB, GPU NVIDIA RTX 3070/4060 Ti (8 GB VRAM) o superior.
    Nota: puede ejecutarse en CPU, pero la velocidad de inferencia se reduce considerablemente.
"""


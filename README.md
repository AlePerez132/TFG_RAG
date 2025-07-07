# GUIA DE INSTALACIÓN Y USO

![Diagrama](TFG_RAG/Diagrama aplicación TFG RAG.jpg)
## INICIALIZACION
Para inicializar la aplicación, es necesario usar ollama, y preferiblemente, desde WSL o directamente Linux, para aprovechar la GPU de NVIDIA, ya que desde windows no se permite.

para configurar wsl, debemos escribir el siguiente comando en una terminal Windows:
```cmd
wsl --install
```

Luego debemos descargar una distribución Ubuntu:
```cmd
wsl --install -d Ubuntu
```

Y entraremos a ella mediante el comando:
```cmd
wsl -d Ubuntu
```

Después, instalaremos Ollama en nuestra distribución Ubuntu:
```cmd
curl -fsSL https://ollama.com/install.sh | sh
```

Y escribiremos el siguiente comando para correr el modelo de lenguaje, en nuestro caso Mistral, que estará disponible en el puerto 11434 porque lo ejecutamos desde Ollama:
```cmd
ollama run mistral
```

Posteriormente ejecutar en la carpeta del proyecto:
```cmd
pip install ./requirements.txt -r
```
para instalar todas las librerias necesarias.

## GPU
Si queremos usar la GPU para acelerar tremendamente el tiempo de respuesta del modelo de lenguaje, debemos instalar el kit de desarollador de Nvidia, el cual se puede descargar desde el siguiente enlace:
https://developer.nvidia.com/cuda-downloads

si todo funciona correctamente, deberíamos poder usar el siguiente comando, aún en la distribución Ubuntu:
```cmd
nvidia-smi
```

para poder visualizar los procesos que usan la GPU. Una vez tengamos Ollama corriendo en nuestra distribución Ubuntu, podemos abrir una nueva terminal, entrar en la distribución Ubuntu, y comprobar que se está ejecutando Ollama. Debería aparecer algo así:
```cmd
nvidia-smi
Tue Apr 22 19:07:41 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.124.04             Driver Version: 572.61         CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060        On  |   00000000:01:00.0  On |                  N/A |
|  0%   41C    P2            N/A  /  115W |    6840MiB /   8188MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A              25      G   /Xwayland                             N/A      |
|    0   N/A  N/A              36      G   /Xwayland                             N/A      |
|    0   N/A  N/A            1189      C   /ollama                               N/A      |
+-----------------------------------------------------------------------------------------+
```

## EMBEDDINGS

Antes de usar el chat por primera vez, es importante crear los embeddings a partir de los documentos a usar.
Para ello, debemos insertar los documentos que necesitemos en la carpeta ./pdf los documentos que necesitemos en formato PDF obligatoriamente.

Una vez se hayan registrado estos documentos, es necesario convertirlos a embeddings a través del comando:
```cmd
python ./embeddings
```
estando situado en la carpeta del proyecto.

Cuando se ejecuta este comando, se crea una carpeta llamada **faiss_index** que contiene 2 archivos, **index.faiss** e **index.pkl**.

Si ya existen estos archivos, es porque ya se han creado los embeddings anteriormente.

## CHAT RAG

Para usar la aplicación del chat de RAG, debemos asegurarnos haber completado todas las instalaciones anteriores, abrir la distribución de Ubuntu de wsl y ejecutar el siguiente comando desde la misma:
```cmd
ollama run mistral
```

Para lanzar el chat de chainlit al navegador, se requiere el siguiente comando en la carpeta del proyecto.
```cmd
chainlit run ./app.py -w
```
La opción -w es para que la aplicación se recargue automáticamente al guardar los cambios del proyecto.

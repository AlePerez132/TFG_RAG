# INICIALIZACION
Para iniciar la aplicación es necesario usar docker, ollama y ejecutar el siguiente comando en CMD:
```cmd
docker run -d -p 11434:11434 --name ollama_container ollama/ollama:latest
```
esto es para lanzar un contenedor con una imagen de ollama
luego al ejecutar la linea 
```cmd
docker ps
```
debería verse la información del contenedor correspondiente:
```cmd
CONTAINER ID   IMAGE                  COMMAND               CREATED         STATUS         PORTS                      NAMES
26199d50e848   ollama/ollama:latest   "/bin/ollama serve"   5 minutes ago   Up 5 minutes   0.0.0.0:11434->11434/tcp   ollama_container
```
si luego al ejecutar el siguiente comando en la **consola de comandos de docker**:
```cmd
ollama list
```
no se ve nada, debemos ejecutar:
```cmd
ollama pull mistral 
```
para descargar el modelo de lenguaje

posteriormente ejecutar en la carpeta del proyecto:
```cmd
pip install ./requirements.txt -r
```
para instalar todas las librerias necesarias.

# EMBEDDINGS

Antes de usar el chat por primera vez, es importante crear los embeddings a partir de los documentos a usar.
Para ello, debemos insertar los documentos que necesitemos en la carpeta ./pdf los documentos que necesitemos en formato PDF obligatoriamente

Una vez se hayan registrado estos documentos, es necesario convertirlos a embeddings a través del comando:
```cmd
python ./embeddings
```
estando situado en la carpeta del proyecto.

Cuando se ejecuta este comando, se crea una carpeta llamada **faiss_index** que contiene 2 archivos, **index.faiss** e **index.pkl**.

# CHAT RAG

Para usar la aplicación del chat de RAG, debemos asegurarnos haber completado todas las instalaciones anteriores, abrir la aplicación de **Ollama**, ejecutar la imagen de ollama en **Docker** y ejecutar el comando 
```cmd
ollama run mistral
```
en la **consola de comandos de Docker**.

Para lanzar el chat de chainlit al navegador, se requiere el siguiente comando.
```cmd
chainlit run ./app.py -w
```
La opción -w es para que la aplicación se recargue automáticamente al guardar los cambios del proyecto.
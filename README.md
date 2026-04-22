# Monitor de Elevacao Lateral

Projeto em Python que usa webcam, MediaPipe e OpenCV para acompanhar a execucao de elevacao lateral em tempo real.

## Tecnologias

- Python 3.11+
- MediaPipe
- OpenCV

## O que o app faz

- detecta ombros, cotovelos e pulsos pela camera;
- identifica o movimento de subida e descida do exercicio;
- espera uma repeticao inicial para calibrar a analise;
- acompanha a trajetoria do pulso durante o movimento;
- marca alertas tecnicos simples, como:
  - pulso acima do cotovelo;
  - ombros encolhidos;
- classifica cada repeticao como correta ou incorreta com base nos frames sem erro;
- mostra a camera com a pose detectada e imprime o resultado de cada repeticao no console.

## Como ele funciona

1. O app abre a webcam e tenta localizar a pose usando o `PoseLandmarker` do MediaPipe.
2. Nos primeiros frames, ele coleta uma referencia da altura media dos ombros.
3. A primeira repeticao completa serve como aquecimento para iniciar o rastreio.
4. Depois disso, cada repeticao passa a ser analisada:
   - quando o pulso sobe acima do limite definido, a repeticao comeca;
   - quando o pulso volta para baixo, a repeticao termina;
   - durante esse trajeto, o app registra erros tecnicos frame a frame.
5. Ao final, o sistema decide se a repeticao foi valida e imprime algo como:
   - `Repeticao 1: correta`
   - `Repeticao 2: incorreta (Ombros encolhidos)`

## Dependencias

As dependencias principais do projeto estao em `requirements.txt`:

- `mediapipe`
- `opencv-contrib-python`
- `numpy`

## Controles

- `Q` ou `ESC`: encerra o app

## Observacoes

- A fase do movimento ainda usa o lado direito como referencia principal para iniciar e encerrar repeticoes.

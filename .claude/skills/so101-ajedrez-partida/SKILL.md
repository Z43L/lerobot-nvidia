---
name: so101-ajedrez-partida
description: Orquesta una partida de ajedrez entre el humano y el SO-101 (sim ahora, real en el futuro). Lleva el turno, valida las jugadas del humano, elige y ejecuta las del robot con so101-ajedrez-mover, y guarda la partida. Úsala cuando el usuario diga "juguemos al ajedrez", "nueva partida", "te toca", "juega tu jugada", "mi jugada es ..." o "estado de la partida".
---

# Partida de ajedrez con el SO-101

Esta skill **orquesta**: decide de quién es el turno, pide o elige la jugada, y deja que `so101-ajedrez-mover` la ejecute. No mueve el brazo por su cuenta.

## Comandos

Todos desde la raíz del proyecto, con `.venv312`. El CLI está en `sim/chess_cli.py`.

| Qué hace | Comando |
|---|---|
| Nueva partida (humano con blancas) | `chess_cli.py new` |
| Nueva partida (robot con blancas) | `chess_cli.py new --robot-white` |
| Ver tablero, turno, legales e historial | `chess_cli.py state` |
| Jugada del humano (SAN o UCI) | `chess_cli.py human <jugada>` |
| Jugada del robot (elige y ejecuta) | `chess_cli.py robot` |
| Imagen de la escena | `chess_cli.py render [fichero.png]` |

Estado de la partida: `hub/chess_game.json` (FEN, historial, color del robot). Si lo borras, se pierde la partida.

## Flujo de una partida

1. **Empezar.** `new` (o `new --robot-white`). Confirma con el usuario quién juega qué color si no lo ha dicho. Muestra `state`.
2. **Turno del humano.** El usuario dice su jugada (por ejemplo "e4" o "Nf3"). Llama a `human <jugada>`.
   - Si la jugada es ilegal, el CLI lo dice y lista las legales. No insistas: pide otra.
   - Si la escena no queda bien (`AVISO`), la jugada no se registra. Explica el aviso y pide al usuario que revise las fichas antes de seguir.
3. **Turno del robot.** Llama a `robot`. Dile al usuario la jugada que ha hecho (SAN y UCI) y el resultado del `steps` (captura, enroque, promoción).
   - Si el robot no puede ejecutar la jugada (`AVISO`), no la des por hecha; muestra `state` y pide ayuda.
4. **Repetir** hasta fin de partida. `state` indica el resultado (mate, tablas) cuando termina.

Antes de cada jugada del robot, si el usuario pide una partida con dificultad distinta, dilo: el rival actual es un minimax de profundidad 2 con material. No es un motor fuerte.

## Reglas

- **Nunca** cambies el FEN a mano ni borres el historial para "arreglar" una partida. Si algo falla, `state` lo refleja; si hace falta reconstruir, hay que dejarlo registrado en el chat.
- **Nunca** ejecutes una jugada del robot sin que sea su turno (el CLI lo bloquea de todos modos).
- **Real:** cada jugada del robot con el brazo real se ejecuta directamente con `so101-ajedrez-real` (ver `so101-ajedrez-partida-real`).
- Lo que no está probado en real se dice: el backend real no existe todavía (`CHESS_BACKEND=real` sale con error a propósito).

## Estado de la prueba (sim)

- Partida de 4 jugadas probada por CLI: `e4`, `g5` (robot), `Nf3`, `g4` (robot), sin avisos de escena.
- Enroque, captura y promoción probados en `ChessSim.execute`; el paso por el CLI no se ha probado para promoción.
- El rival es material con profundidad 2: juega mal a propósito por ahora. Cambiarlo es fácil (`sim/chess_cli.py`, `engine_move`).

## Cambiar el rival o el backend en el futuro

- **Rival:** `engine_move` en `sim/chess_cli.py`. Se puede sustituir por Stockfish u otro motor sin tocar esta skill.
- **Backend real:** ver el apartado "Paso a real" de `so101-ajedrez-mover`. Esta skill no cambia.

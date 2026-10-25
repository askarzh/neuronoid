#!/usr/bin/env bash
# Наблюдение за прогоном: строка раз в 25 минут, сигнал на каждом новом миллионе шагов и в конце.
# Запуск: bash scripts/watch_run.sh ppo2
run=$1
cd "$(dirname "$0")/.."
log=runs/$run.log
ckpt=runs/$run/checkpoints/latest.json

million() {  # миллионы шагов последнего чекпоинта; при гонке с заменой файла — прошлое значение
  local v
  v=$(tr -dc 0-9 < "$ckpt" 2>/dev/null)
  if [ -n "$v" ]; then echo $((v / 1000000)); else echo "${1:-0}"; fi
}

last=$(million 0)
echo "$run: наблюдение включено, чекпоинт на ${last} млн"
n=0
while true; do
  sleep 5
  n=$((n + 1))
  if grep -qE "Сохранено|Traceback|Нет шагов" "$log"; then
    echo "$run END: $(grep -E 'Сохранено|Error|Нет шагов' "$log" | tail -2 | tr '\n' ' ')"
    exit 0
  fi
  m=$(million "$last")
  if [ "$m" -gt "$last" ]; then
    last=$m
    echo "$run CHECKPOINT ${m}000000 ready for eval"
  fi
  if [ $n -ge 300 ]; then
    n=0
    echo "$run $(date +%H:%M) steps=$(grep total_timesteps "$log" | tail -1 | tr -dc 0-9)"
  fi
done

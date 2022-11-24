

PLIB_PIDS=()

function prun {
  $@ &
  PLIB_PIDS+=($!)
}

function add_pid() {
  PLIB_PIDS+=($!)
}

pwait() {
  ## Wait for children to exit and indicate whether all exited with 0 status.
  local iter_pids=("${PLIB_PIDS[@]}")
  PLIB_PIDS=()
  while (("${#iter_pids[@]}" > 0)); do
    # echo "Processes remaining: ${#iter_pids[@]} - ${iter_pids[@]}"
    for pid_idx in "${!iter_pids[@]}"; do
      pid="${iter_pids[pid_idx]}"
      if kill -0 "$pid" 2>/dev/null; then
        # "$pid is still alive."
        :
      elif wait "$pid"; then
        # "$pid exited with zero exit status."
        unset 'iter_pids[pid_idx]'
      else
        status=$?
        # "$pid exited with non-zero exit status."
        unset 'iter_pids[pid_idx]'
        # kill all remaining processes
        for pid in "${iter_pids[@]}"; do
          if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
          fi
        done
        return $status
      fi
    done
    sleep 0.1
   done
}

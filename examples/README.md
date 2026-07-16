# Example trajectories

`trajectory_A.csv` and `trajectory_B.csv` describe the same diagonal peel with
different early Z heights. They are deliberately synthetic and exist only to
exercise the comparator; they are not process recommendations.

The CSV contract is:

```text
point,x_mm,y_mm,z_mm,speed_mm_s
```

- Coordinates are the pull-tape grip point in the panel-fixed frame.
- P1 is the fully attached start state and P6 is the completed-peel state.
- In the default speed mode, the speed on Pi applies to Pi -> Pi+1; P6 speed is
  preserved but unused.


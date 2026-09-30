// Episodes and training logs recorded by scripts/train_mujoco_ppo.py. Add a task here after training it.
import ant from "./ant.json";
import halfCheetah from "./half_cheetah.json";
import hopper from "./hopper.json";
import invertedPendulum from "./inverted_pendulum.json";
import pointMaze from "./point_maze.json";

export const MUJOCO_TASKS = [
  {
    id: "inverted_pendulum",
    title: "InvertedPendulum",
    robot: "a cart that must keep a pole upright. Reward +1 for every step the pole has not fallen, up to 1000",
    data: invertedPendulum,
  },
  {
    id: "hopper",
    title: "Hopper",
    robot: "a one-legged robot that must hop forward without falling. Reward: forward speed plus +1 for every step it stays upright, minus a small cost for large torques; the episode ends when it falls, or after 1000 steps",
    data: hopper,
  },
  {
    id: "half_cheetah",
    title: "HalfCheetah",
    robot: "a two-legged robot in a vertical plane that must run forward. Reward: forward speed minus a small cost for large torques. The replay shows the first 400 of the 1000 steps of an episode",
    data: halfCheetah,
  },
  {
    id: "ant",
    title: "Ant",
    robot: "a four-legged robot that must walk forward. Reward: forward speed plus +1 for every step it stays upright, minus a cost for large torques; the episode ends if it flips over. Standing still already earns about +1 per step, which is why the untrained policy scores well. Trained for 2M steps without the 78 contact-force numbers in the observation: with them, PPO did not learn to walk in 3M steps. The replay shows the first 400 steps",
    data: ant,
  },
  {
    id: "point_maze",
    title: "PointMaze",
    robot:
      "a ball that must roll to the green goal in the U-shaped maze, the continuous version of the grid lessons. Reward: exp(−distance to the goal) every step, and the episode ends at the goal. Look at episode length, not return: reaching the goal sooner ends the episode sooner, so a faster policy collects less reward than one that hovers nearby. PPO found the goal anyway, but this is a real reward-design trap",
    data: pointMaze,
  },
];

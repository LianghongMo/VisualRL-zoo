// The main tutorial path from the README. Lessons with `ready` have an interactive page.
export const PARTS = [
  {
    part: "Part I",
    title: "What is reinforcement learning?",
    lessons: [
      { id: "01", title: "Agent and Environment", about: "State, action, reward, transition, episode, trajectory." },
      { id: "02", title: "Reward, Return, and Discounting", about: "Drag γ and watch a near small reward beat a far large one.", ready: true },
      { id: "03", title: "Policies and Trajectories", about: "One policy, many trajectories, a visitation heatmap." },
      { id: "04", title: "Exploration and Exploitation", about: "Greedy vs ε-greedy on a multi-armed bandit." },
    ],
  },
  {
    part: "Part II",
    title: "Learning values",
    lessons: [
      { id: "05", title: "State Value and Action Value", about: "V(s) in every cell, Q(s,a) in every direction." },
      { id: "06", title: "Bellman Backup", about: "Expand one backup into its branches and compute it by hand." },
      { id: "07", title: "Policy Evaluation and Policy Improvement", about: "Two separate buttons that change two separate things." },
      { id: "08", title: "Monte Carlo vs TD", about: "The same trajectory, two targets, and n-step TD in between.", ready: true },
    ],
  },
  {
    part: "Part III",
    title: "Control",
    lessons: [
      { id: "09", title: "SARSA vs Q-learning", about: "Behavior action vs target action on the cliff.", ready: true },
      { id: "10", title: "On-policy, Off-policy, and Replay", about: "A visible replay buffer and the minibatches drawn from it." },
    ],
  },
  {
    part: "Part IV",
    title: "Function approximation",
    lessons: [{ id: "11", title: "From Tables to Functions", about: "Update one state and watch predictions move everywhere." }],
  },
  {
    part: "Part V",
    title: "Deep reinforcement learning",
    lessons: [
      { id: "12", title: "DQN", about: "Replay, online network, target network, Bellman target, loss." },
      { id: "13", title: "REINFORCE", about: "Logits to probabilities to a policy gradient." },
      { id: "14", title: "Actor-Critic and Advantage", about: "Update the critic, then update the actor." },
      { id: "15", title: "PPO", about: "Probability ratio, advantage, and the clipped objective." },
      { id: "16", title: "Evaluation and Debugging", about: "Seeds, uncertainty, and why a falling loss is not a better policy." },
    ],
  },
];

export const LESSONS = PARTS.flatMap((p) => p.lessons.map((l) => ({ ...l, part: p.part, partTitle: p.title })));
export const lessonById = (id) => LESSONS.find((l) => l.id === id);

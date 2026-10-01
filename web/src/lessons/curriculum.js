// The main tutorial path. Lessons with `ready` have an interactive page.
export const PARTS = [
  {
    part: "Part I",
    title: "The problem and the loop",
    lessons: [
      { id: "01", title: "The World as a Graph", about: "States are nodes, moves are edges, rewards are weights; optimal control is a best path.", ready: true },
      { id: "02", title: "The Learning Loop", about: "Acting adds edges to the experience graph; learning moves value back along them.", ready: true },
      { id: "03", title: "Online and Offline", about: "Stitching and coverage offline, exploration online, and what an untried move is worth.", ready: true },
    ],
  },
  {
    part: "Part II",
    title: "Learning values",
    lessons: [
      { id: "04", title: "State Value and Action Value", about: "Fix a policy: V(s) on every node, Q(s,a) on every edge, and where Q beats V.", ready: true },
      { id: "05", title: "Bellman Backup and the Optimal Policy", about: "One backup by hand, then repeated everywhere until the optimal policy appears.", ready: true },
      { id: "06", title: "Policy Iteration", about: "Evaluate and improve: two buttons that change two different things.", ready: true },
      { id: "07", title: "Monte Carlo vs TD", about: "The same trajectory, two targets, and n-step TD in between.", ready: true },
    ],
  },
  {
    part: "Part III",
    title: "Control",
    lessons: [
      { id: "08", title: "SARSA vs Q-learning", about: "A warehouse robot at a loading ledge: behavior action vs target action, and an experience graph.", ready: true },
      { id: "09", title: "On-policy, Off-policy, and Replay", about: "A visible replay buffer and the minibatches drawn from it." },
    ],
  },
  {
    part: "Part IV",
    title: "Function approximation",
    lessons: [{ id: "10", title: "From Tables to Functions", about: "Update one state and watch predictions move everywhere." }],
  },
  {
    part: "Part V",
    title: "Deep reinforcement learning",
    lessons: [
      { id: "11", title: "DQN", about: "Replay, online network, target network, Bellman target, loss." },
      { id: "12", title: "REINFORCE", about: "Logits to probabilities to a policy gradient." },
      { id: "13", title: "Actor-Critic and Advantage", about: "Update the critic, then update the actor." },
      { id: "14", title: "PPO", about: "Rollouts, GAE, probability ratios and clipping, in the warehouse and on MuJoCo robots.", ready: true },
      { id: "15", title: "Evaluation and Debugging", about: "Seeds, uncertainty, and why a falling loss is not a better policy." },
    ],
  },
];

export const LESSONS = PARTS.flatMap((p) => p.lessons.map((l) => ({ ...l, part: p.part, partTitle: p.title })));
export const lessonById = (id) => LESSONS.find((l) => l.id === id);

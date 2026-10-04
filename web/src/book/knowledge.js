export const CHAPTER_KNOWLEDGE={
  "01":[["状态与MDP","task","为什么状态需要包含速度？"],["G、V、Q与π*","values","G、V和Q分别评价什么？"],["Bellman方程","bellman","怎样从return分解得到Qπ、Vπ和最优方程？"],["Bellman与Q-learning","bellman","样本reward和估计的后续Q怎样组成学习目标？"],["动态规划","planning","策略评价、策略改进和价值迭代各更新什么？"]],
  "02":[["数据访问权限","access","能否采新数据与能否任意查询状态，是同一个问题吗？"],["采样误差","statistics","样本数和折扣怎样影响误差？"],["MC、TD、SARSA与Q-learning","updates","它们使用哪种未来目标？"],["终止与截断","updates","什么时候保留后继V？"]],
  "03":[["价值特征","features","线性特征能否表示V(x)=−x²？"],["表示与闭包","closure","能表示Q*，为何还需要Bellman闭包？"],["FQI与覆盖","offline","没有采过的动作，能靠回归判断吗？"],["Stitching","stitching","位置相同、速度不同的片段能拼接吗？"]],
  "04":[["UCB","bandit","均值与不确定性怎样决定动作？"],["策略性探索","strategic","怎样到达有学习信号的状态？"],["Linear MDP","structure","它与线性机械动力学有何区别？"],["Bellman rank","structure","哪个残差矩阵需要低维分解？"]],
  "05":[["Gaussian策略","gaussian","正优势怎样改变动作分布？"],["PG推导","derivation","为什么去掉过去reward仍保留外层γᵗ？"],["Baseline","baseline","为什么不改期望，却能改方差？"],["Actor–Critic与GAE","actor-critic","终止和截断怎样影响GAE？"]],
  "06":[["驻点与最优策略","optimality","小梯度为何不一定意味着最优？"],["KL与Fisher","geometry","方差如何影响同一个均值步长的KL？"],["自然梯度与兼容近似","compatible","score回归怎样得到Fisher方向？"]],
  "07":[["性能差与surrogate","surrogate","surrogate使用哪个状态分布？"],["CPI与TRPO","trust-region","平均KL小，是否意味着每个状态变化都小？"],["PPO clip","ppo","正负优势分别在哪一侧截断？"],["Jensen与熵","entropy","soft目标的KL差距何时为0？"],["更新流程","practice","一个批次内哪些量必须固定？"]],
  "08":[["BC损失","bc","BC与RL分别使用什么监督？"],["分布偏移","shift","动作回放为何无法应对扰动？"],["交互模仿","interactive","专家动作与专家Q分别支持什么方法？"],["逆强化学习","reward","示范能否唯一确定reward？"]],
  "09":[["动力学","dynamics","推力怎样更新位置和速度？"],["二次reward","objective","增大R怎样改变策略？"],["Riccati","riccati","如何从Bellman得到P、K和u=−Kx？"],["价值与轨迹","feedback","累计return是否等于V₀*？"],["稳定性与限幅","limits","有限时域最优能否保证渐近稳定？"]],
  "10":[["环境接口","interface","观测、动作、reward与终止怎样定义？"],["连续策略","algorithms","tanh的Jacobian修正是什么？"],["目标条件任务","gcrl","换目标后，策略和reward怎样变？"],["HER与未来占用","relabel","改目标后怎样重算reward与终止？"],["实验评估","evaluation","怎样检验训练结果与目标泛化？"]],
};

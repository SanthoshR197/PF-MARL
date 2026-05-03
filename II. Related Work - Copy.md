**II. Related Work**



The existing literature on using machine learning for epidemic control covers several interconnected domains, such as reinforcement learning for policy decisions, multi-agent cooperation, simulation-based disease modeling, and privacy-preserving distributed learning.To systematically position our work, we categorize prior studies into four key directions: (i) reinforcement learning approaches for epidemic control, (ii) multi-agent and advanced reinforcement learning frameworks, (iii) simulation and hybrid epidemiological modeling techniques, and (iv) privacy and federated learning-based methods. This structured segregation enables a clear understanding of the capabilities and limitations of existing approaches and highlights the gap addressed by our proposed PF-MARL framework.



**A. Reinforcement Learning for Epidemic Control**



Reinforcement learning (RL) has become a promising paradigm for improving intervention plans in disease outbreaks because it works well with sequential decision making under uncertain conditions. Early research mainly centered on finding the best actions inside centralized epidemic modeling frameworks. 



Libin et al. \[3] built a deep RL system that runs on a large-scale SEIR model with hundreds of interconnected regions. Their method used policy gradient techniques to learn effective mitigation strategies such as school closure policies. The study showed that coordination between districts leads to better health results. Still, the training process runs from one central point and doesnt handle decisions made independently or protect personal data.



Similarly, Bushaj et al. \[5] developed a simulation-Deep Reinforcement Learning (SiRL) setup that combines agent-based epidemic modeling with a deep RL decision-maker. This method allows testing various interventions like vaccine allocation and policy limits using a reward system that measures both health outcomes and economic effects. And the system depends on a single central agent and does not support coordination between different geographic areas. And the model assumes all regions follow the same rules, which may not reflect real-world scenarios.



Khatami and Gopalappa \[8] expanded RL-based epidemic control by adding interactions between multiple juristrictions. Their model treats epidemic management as a markov Decision Process and uses deep Q-networks to improve lockdown plans that balance reducing infections, deaths, and economic costs. The model includes travel patterns to show how regions affect each other,yet it only works with smaller regions and cant handle broader, more complex setups.



Overall, these approaches demonstrate the effectiveness of reinforcement learning in epidemic control. Still, they mostly rely on centralized learning paradigms, dont support scalable coordination among many agents, and o not incorporate privacy-preserving mechanisms. This makes them less useful in actual large-scale situations where data privacy matters.



B. Multi-Agent and Advanced Reinforcement Learning Approaches



To overcome the limitations of centralized reinforcement learning, recent studies have explored advanced formulations incorporating multi-agent coordination, hierarchical decision-making, and graph-based learning to better model the distributed and interconnected nature of epidemic systems.





The framework developed by Luo et al.\[9], called H2-MARL,a multi-agent reinforcement learning framework designed to achieve Pareto optimality between conflicting objectives, creates separate agents for each administrative area and uses a dual-goal reward system to adjust both health outcomes and economic impacts over time. It connects to a spatiotemporal epidemic simulator built on a dynamic SIHR model with updated parameters and includes a large human mobility dataset containing over one billion records from various cities. Results show the method reduces strain on hospitals and limits disruptions from mobility rules although still working well across different urban settings. Still, the system needs centralized training and full access to private health and movement data. It makes it hard to use in places where privacy rules restrict data sharing.



Du et al. \[10] developed HRL4EC, a hierarchical reinforcement learning frameowrk for managing outbreaks across different control modes. It breaks down complicated intervention plans into layered decision-making steps. The method uses a novel epidemiological model called MID-SEIR to clearly show how various actions affect infection spread. By reorganizing the outbreak management challenge into a hierarchical structure , the system simplifies the decision space and supports improving several interventions at once. Testing on real and synthetic data shows better results than methods using just one intervention approaches.



Hurtado and marculescu \[11] came up with a graph-based system for multi-agent reinforcement learning that blends Graph Neural Networks with MARL to help manage disease spread in a decentralized way. The method forecasts infection chances at specific locations using actual movement data and runs many agents to adjust how they move based on those predictions. It uses both spatial and movement patterns to spot how people interact. Now, this helps lower the reproduction rate below important levels without needing hard lockdowns. Despite its strong alignment with real-world urban dynamics, the framework assumes access to detailed individual-level mobility data and does not incorporate formal privacy guarantees or federated learning mechanisms.



These approaches seem promising in modeling how diseases spread across networks and supporting group actions. But most depend on a central source for data,  lack integration with federated learning paradigms, and do not provide rigorous privacy guarantees, thereby limiting their deployment in real-world, privacy-sensitive urban environments.



C. Simulation and Epidemiological Modeling Techniques



Simulation-based and epidemiological modeling approaches play a crucial role in understanding disease transmission dynamics and evaluating intervention strategies. These methods provide structured representations of population interactions, mobility patterns, and disease progression, forming the foundation for intelligent decision-making frameworks.



By portraying people as independent entities with unique behaviours, agent-based modelling (ABM) has been widely used to capture fine-grained epidemic dynamics. In order to assess the risks of COVID-19 transmission within facilities, Cuevas \[16] proposed an agent-based framework in which each agent adheres to predetermined behavioural rules that are influenced by individual characteristics and spatial interactions. This method makes it possible to model diverse populations and accurate micro-level transmission patterns. In a similar vein, Truszkowska et al. \[19] created a high-resolution agent-based model that could replicate the spread of disease at the individual agent level in an actual urban environment. Their model includes specific demographic, spatial, and behavioural characteristics, making it possible to assess intervention tactics like testing and immunisation with extreme accuracy.



Network-based epidemiological approaches concentrate on interactions between aggregated population groups, as opposed to individual-level modelling. A metapopulation network model was presented by Humphries et al. \[17], where regions are represented as nodes connected by mobility links. While travel and commuting patterns are captured by inter-node connections, each node tracks compartmental epidemic dynamics. This structure offers a natural representation for graph-based epidemic systems and makes it possible to model large-scale disease transmission across geographically dispersed regions.



Furthermore, the significance of human mobility in influencing transmission dynamics has been shown by mobility-driven epidemic models. A hybrid gravity-metapopulation model that incorporates spatial distance and mobility data into the transmission process was proposed by Iyaniwura et al. \[18]. Their framework allows for the assessment of how interregional movement affects infection rates and takes into account time-dependent mobility patterns.



Overall, These studies offer different views on how epidemics spread, from tracking single people to modeling entire networks. Some focus on precise simulations but lack smart decisions over time. Others include only a bit of coordination between areas. Learning-based improvements arent well combined with privacy-friendly, distributed systems. These limitations motivate the need for advanced approaches that combine simulation fidelity, multi-agent coordination, and adaptive policy learning, as addressed in the proposed PF-MARL framework.



D. Privacy-Preserving and Federated Learning Approaches



The increasing reliance on data-driven epidemic control systems  brings concerns about privacy, security, and distributed collaborations. Epidemic data is personal, spread across different government areas, and protected by strict regulatory constraints. As a result, traditional centralized learning approaches that require raw data aggregation are often impractical. To address these limitations, recent research has explored federated learning and privacy-preserving reinforcement learning frameworks.



Federated learning lets different groups train a shared model without sharing raw data. This method helps keep information private at the same time still allowing learning to happen across locations.  Zhou et al. \[22] created a system using federated reinforcement learning to make decisions during epidemics by combining regional data without exposing sensitive details. The results show this setup reaches decisions faster and performs better than single-region models, Mainly when data is scarce and delicate. However, the approach remains largely centralized in its coordination mechanism and does not fully address scalability and heterogeneity across regions.



Antunes et al. \[23] look at how federated learning works in healthcare, showing it helps run machine learning on sensitive data like electronic health records. The method lets models train locally at each site and only sends model parameters, which keeps patient data private and follows privacy regulations.Despite its advantages, federated learning faces challenges such as communication overhead, data heterogeneity, and vulnerability to inference attacks.

To improve scalability and robustness, decentralized federated learning approaches have been proposed. Guan et al. \[24] developed a decentralized model based on population movement patterns, where nodes share updates directly without using a central server. Their system uses a mobility-based adjacency matrix to track connections between regions. It also keeps individual nodes separate and private.This structure is particularly relevant for epidemic modeling, where disease spread is inherently influenced by mobility-driven interactions.



In parallel, privacy-preserving distributed learning methods aim to protect data and model details during group training Froelicher et al. \[26] introduced SPINDLE, a system that works across multiple data sources using homomorphic encryption so participants can train together without sharing sensitive data. This method keeps both data and model details private, even when someone tries to attack the process, and delivers results similar to central training systems.However, such cryptographic methods often introduce computational overhead and may be difficult to integrate with adaptive learning frameworks like reinforcement learning.



Differential privacy offers a formal way to shield individual data during model training. Yang-Zhao and ng \[21] studied privacy in reinforcement learning within population models, adding random noise to state and reward inputs to meet privacy standards. Their work demonstrates that it is possible to achieve a favorable trade-off between privacy and utility, particularly in large-scale population settings, where approximation errors decrease as the population size increases  Yet, applying differential privacy in reinforcement learning brings extra issues like higher randomness and possible drops in policy performance.



Overall, even though present studies have achieved substantial progress in federated learning, distributed systems, and differential privacy, they mostly consider these elements separately. The methods in use now either emphasize federated coordination but lack stringent privacy provisions, or focus on privacy-preserving technologies without considering adaptive multi-agent decision-making scenarios. Besides, the combined issues of upgrading, communication effectiveness, and decentralized policy learning are still hardly touched upon.



These limitations motivate the development of unified frameworks that integrate federated learning, multi-agent reinforcement learning, and differential privacy into a cohesive system. The proposed framework PF-MARL, closes the existing gap by facilitating multiple-region decentralized policy optimization with strong privacy assurance and efficient coordination for large-scale epidemic control problem-solving.



TABLE I: Comparative Summary of Related Works



Table 1 illustrates the major merits and demerits of different approaches in reinforcement learning, multi-agent systems, epidemiological modeling, and privacy-preserving distributed learning. Even though reinforcement learning methods can highly optimize intervention strategies, most of the time, they are focused on centralized settings only and do not have the ability to coordinate multiple regions on a large scale. Multi-agent reinforcement learning approaches mitigate this problem to some extent by allowing distributed decision-making; however, they typically assume unrestricted access to sensitive data and do not incorporate formal privacy guarantees.



Simulation-based and epidemiological models offer an accurate depiction of disease spread and human mobility but have hardly changed their operation to make the policy actually be the result of a learning process. On the other hand, federated learning and privacy-preserving methods solve the issue of data control by a single party through decentralized training and a formal privacy guarantee however their operation is quite the opposite to that of sequential decision-making or coordinated control in the course of a dynamic epidemic environments.



As a result, existing approaches fail to simultaneously achieve scalable multi-agent coordination, adaptive policy learning, and rigorous privacy preservation within a unified framework. This limitation motivates the proposed PF-MARL system, which integrates multi-agent reinforcement learning with federated learning and differential privacy to enable decentralized, privacy-aware, and dynamically optimized epidemic control across interconnected urban regions.








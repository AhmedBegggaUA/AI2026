# The Project
 A The **Discrete Brain** is an educational initiative oriented to explore the interconnection between discrete and continuous maths for inspiring algorithms and even to *neuralize* them. The long-term objective is to foster a principled development of Artificial Intelligence (AI). 

In our opinion, the AI-Engineer (AIE) should master discrete maths, including probability as a bridge to heuristic solvers of NP-hard problems. This is the subject of **Distrinite Mathematics** (Discrete maths. or DM, for AI), where we set the basis of alogithmic exploration. 

In DM, we address our first **NP-Hard** problem, namely MinCut, where we must find the best balanced partition in a graph. What is relevant to an AIE is at this point we realize that NP-Hard problems have to be **relaxed** or approximated by a polynomial surrogate. 

It is in **Herisstics** (Advanced Search for AI, or H), where NP-Hard problems are characterized and some approximations of "relaxations" are suggested. In this regard, more conventional state-based search approaches (such as $A^{\ast}$ and Game Search) are adressed therein. In particular, the concept of **heuristic** (a good discriminator of uniformative states) emerges naturally. However, in modern AI, heuristics are no longer mathematical rules but "deep oracles" or Neural Networks or NN. 

Assuming that a NN is always available (trained beforehand), problems like the Rubik Cube can be posed in terms of using the NN to navigate through the huge state space (DeepCube). However, the contribution of the NN is basically "local" as we reflect in its application to Alpha-Beta search in Games. It is the embedding of this NN in a systematic search which makes it "globally powerful". 

Then, with DM and H, we are prepared to design **Intelligent Agents** (this subject). Here they are the three special modules. 

**1) Algoritmic Neuralization**. Future algorithms will be "code independent", i.e. they can be in turn <u>approximated by a NN</u> aligned with the procedural or functional flavor of the algorithm (e.g. sorting). Recent developments in the subject show that this can be done not only by using Transformers (the flagship NN model for AI) but also by means of **Graph Neural Networks** or GNNs. We will study these advanced models which leverage graphs (a recurrent topic in both the DM and the H subjects). In particular, GNNs encode algorithms with a graph whose edges and node's attributes are learnt by specifying the cost function (e.g. the distance to the correct permutation in sorting). 

**2) Reinforcement Learning**. This is the second block  of our subject. Herein <u>we move from an algorithm to an agent</u>. In short, the first ingredient of an agent is to "interact with the environment". Interactions provide information to learn from. In addition, agents do have a "reasoning algorithm". Typically, RL relies on learning strategies to link interactions and reasoning so that we can generalize game playing, mathematical reasoning, and so on, in line with the [Artificial General Intelligence (AGI)](https://en.wikipedia.org/wiki/Artificial_general_intelligence) dream. This second part is more mathematical and it requires to refresh our concepts on Markov chains and random walks, and Dynamic Programming. Actually, the game-search part is a continuation of what we have learnt is the H subject. 

**3) LLMs and VLMs**. The third ingredient is the programing and design of Inteligent Agents (IAGs). The dominant approach here is to exploit yet learnt LLMs as well as Visual Language Models (VLMs) via RL. A nice example is to use an LLM to train an agent which is capable of doing certain tasks (not just a pipeline but a RL agent). One of the most intriguing tasks is "mathematical reasoning": <u>Can we train an agent to solve mathematical problems?</u> The answer is Yes. We can also train agents to solve "visual tasks" such as summarizing a scene, for a blind or visually impaired subjet, with a controlled level of verbosity. 

Finally, we close the loop with graphs again since frameworks such as [LangGraph](https://www.langchain.com/langgraph) provide a flexible orchestration for multiple IAGs. 


**Real cases**. A key element of the Discrete Brain is to illustrate the working of agents on realistic cases and this is the main focus of this subject. 


**Team**. The creative team of this project is composed by professors and PhD students with a strong knowledge of the state-of-the-art of structural methods in AI. 

- **Francisco Escolano**. Full professor and coordinator of the project and of this subject. 
- **Alvaro Díez Díaz**. PhD and expert in Reinforcement Learning.


<style>
.circle {
 border-radius: 50%;
 height: 100px;
 width: 100px;
 overflow: hidden;
 display: flex;
 justify-content: center;
 align-items: center;
}
.foto {
 margin-right: 50px; /* Ajusta este valor según tus necesidades */
}
</style>

<div style="display: flex;">
    <div class="foto">
        <div class="circle">
        <img src="https://www.encuentrosnow.es/wp-content/uploads/francisco-escolano.jpg" alt="" />
        </div>
    <p> Francisco Escolano Ruíz </p>
    </div>
    <div class="foto">
        <div class="circle">
        <img src="https://media.licdn.com/dms/image/v2/D4D03AQEv23xwQZRl6A/profile-displayphoto-shrink_200_200/profile-displayphoto-shrink_200_200/0/1669300428222?e=2147483647&v=beta&t=3Le6bN7_YWAJuoGx9gx_7wGO3I_SZ1gHyhUFmwBQ98M" alt="" />
        </div>
    <p> Alvaro Díez Díaz  </p>
    </div>
</div>


# EdgeCVS: Democratization of Surgical AI with a Distilled Edge-Deployable Critical View of Safety (CVS) Model

**MICCAI 2026** · Amine Yamlahi, Jakob Hennighausen, Pascal Hansen, David Leeb, Lena Maier-Hein


Division of Intelligent Medical Systems (IMSY), German Cancer Research Center (DKFZ), Heidelberg

> 🚧 **Code and model weights coming soon.** 

EdgeCVS distills a 305M-parameter EVA02-Large teacher into a **5M-parameter EdgeNeXt-Small** student for
Critical View of Safety assessment in laparoscopic cholecystectomy. The teacher pseudo-labels 131K frames
taken only from the Calot-triangle-dissection phase. Trained on those frames, the student reaches
**66.5 mAP** on the SAGES-CVS 2024 test set (teacher: 66.9) and runs at **40 FPS on a CPU**.

<p align="center">
  <img src="figures/edgecvs_frontier.png" width="100%" alt="CVS mAP versus model parameters on the SAGES-CVS 2024 leaderboard">
</p>



from typing import Dict, Any, List, Union, Optional, Tuple, Iterable
import os
import json
from pathlib import Path
import re
import shutil
import threading
import math

try:
    import openai
except Exception:
    openai = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from PIL import Image
except Exception:
    Image = None

from .graph_memory import GraphMemory
from .retrieval import retrieve_from_graph, collect_memories


class Memory:
    _global_instance: Optional['Memory'] = None  # 单例实例（进程内全局共享）

    def __init__(self, max_memory_length: int = 3, is_enable: bool = True):
        self._memory_root = Path("tmp/memory_store")
        self._memory_root.mkdir(parents=True, exist_ok=True)
        
        self.is_enable = is_enable

        # 防止直接实例化，必须通过 get_instance()
        if Memory._global_instance is not None:
            raise RuntimeError("Use Memory.get_instance() to get the global Memory instance.")
        self.query: Optional[str] = None
        self.files: List[Dict[str, str]] = []
        self.actions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.max_memory_length = max_memory_length
        self._next_clip_id = 0
        self._graph = GraphMemory()
        self._clip_images: Dict[int, List[str]] = {}
        self._last_retrieval: Dict[str, Any] = {}
        self._embedding_backend = os.environ.get("EMBEDDING_BACKEND", "openai").lower()
        self._embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
        self._image_embedding_model = os.environ.get("LOCAL_IMAGE_EMBEDDING_MODEL", "clip-ViT-B-32")
        self._openai_client = None
        self._local_embedder = None
        self._local_image_embedder = None
        self._generator_llm = None
        self._generator_enabled = False
        self._init_file_types()
        print("✅ Initialized global shared Memory (single instance for the entire process)")

    @classmethod
    def get_instance(
        cls,
        max_memory_length: int = 3,
        is_enable: bool = True
    ) -> 'Memory':
        if cls._global_instance is None:
            cls._global_instance = cls(
                max_memory_length=max_memory_length,
                is_enable=is_enable
            )
        return cls._global_instance

    def set_query(self, query: str) -> None:
        if not isinstance(query, str):
            raise TypeError("Query must be a string")
        self.query = query

    def _init_file_types(self):
        self.file_types = {
            'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp'],
            'text': ['.txt', '.md'],
            'document': ['.pdf', '.doc', '.docx'],
            'code': ['.py', '.js', '.java', '.cpp', '.h'],
            'data': ['.json', '.csv', '.xml'],
            'spreadsheet': ['.xlsx', '.xls'],
            'presentation': ['.ppt', '.pptx'],
        }
        self.file_type_descriptions = {
            'image': "An image file ({ext} format) provided as context for the query",
            'text': "A text file ({ext} format) containing additional information related to the query",
            'document': "A document ({ext} format) with content relevant to the query",
            'code': "A source code file ({ext} format) potentially related to the query",
            'data': "A data file ({ext} format) containing structured data pertinent to the query",
            'spreadsheet': "A spreadsheet file ({ext} format) with tabular data relevant to the query",
            'presentation': "A presentation file ({ext} format) with slides related to the query",
        }

    def _get_default_description(self, file_name: str) -> str:
        _, ext = os.path.splitext(file_name)
        ext = ext.lower()

        for file_type, extensions in self.file_types.items():
            if ext in extensions:
                return self.file_type_descriptions[file_type].format(ext=ext[1:])

        return f"A file with {ext[1:]} extension, provided as context for the query"
    
    def add_file(self, file_name: Union[str, List[str]], description: Union[str, List[str], None] = None) -> None:
        if isinstance(file_name, str):
            file_name = [file_name]
        
        if description is None:
            description = [self._get_default_description(fname) for fname in file_name]
        elif isinstance(description, str):
            description = [description]
        
        if len(file_name) != len(description):
            raise ValueError("The number of files and descriptions must match.")
        
        for fname, desc in zip(file_name, description):
            self.files.append({
                'file_name': fname,
                'description': desc
            })

    def reset(self):
        self.query = None
        self.files.clear()
        self.actions.clear()
        self._graph = GraphMemory()
        self._clip_images.clear()
        self._last_retrieval = {}
        self._next_clip_id = 0
        self._generator_llm = None
        self._generator_enabled = False
        self._local_embedder = None
        self._local_image_embedder = None

        if self._memory_root.exists():
            for p in self._memory_root.glob("*"):
                if p.is_dir():
                    for f in p.glob("*"):
                        f.unlink()
                    p.rmdir()
                else:
                    p.unlink()

        print("🧹 Memory reset + internal image store cleared")


    def add_embodied_action(
        self, 
        belief: str,
        intention: str,
        state: str, 
        verification: str,
        commands: List[Tuple[str, str]] = None,
        image_paths: Any = None,
        raw_planner_output: Optional[str] = None,
        task_context: Optional[str] = None,
        interaction_memory: Optional[str] = None,
        execution_time: Optional[float] = None
    ) -> None:

        step_count = len(self.actions) + 1
        step_name = f"Action Step {step_count}"

        action = {
            'interaction_memory': interaction_memory if interaction_memory else None,
            'belief': belief,
            'intention': intention,
            'commands': commands if commands else None,
            'state': state,
            'execution_time': execution_time,
            'verification': None   # wait for verfier to update
        }

        with self._lock:
            self.actions[step_name] = action
            self._add_multimodal_memory(
                belief=belief,
                intention=intention,
                state=state,
                verification=verification,
                commands=commands,
                interaction_memory=interaction_memory,
                image_paths=image_paths,
                raw_planner_output=raw_planner_output,
                task_context=task_context
            )

        return step_name

    def get_total_steps(self) -> int:
        if self.is_enable == False:
            return None
        return len(self.actions)

    def get_query(self) -> Optional[str]:
        if self.is_enable == False:
            return None
        return self.query

    def get_files(self) -> List[Dict[str, str]]:
        if self.is_enable == False:
            return None

        return self.files
    
    def get_actions(self) -> Dict[str, Any]:
        if self.is_enable == False:
            return None
        total_steps = len(self.actions)
        
        all_steps = sorted(self.actions.items(), key=lambda x: int(x[0].split()[-1]))
        recent_steps_list = all_steps[-self.max_memory_length:]
        recent_actions = dict(recent_steps_list)

        return {
            "total_steps": total_steps,
            "memory_window_size": self.max_memory_length,
            "actions": recent_actions,
            "retrieved_memories": self._last_retrieval or None
        }

    def refresh_retrieval_context(self, query: str, topk: int = 3) -> None:
        if self.is_enable == False:
            return
        if not query:
            self._last_retrieval = {}
            return
        self._last_retrieval = self.retrieve_context(query=query, topk=topk)

    def get_memory_images(self) -> List[str]:
        if self.is_enable == False:
            return []
        images = []
        retrieved = self._last_retrieval.get("clips", []) if self._last_retrieval else []
        for clip in retrieved:
            images.extend(clip.get("images", []))
        return images

    def retrieve_context(self, query: str, topk: int = 3) -> Dict[str, Any]:
        if self.is_enable == False:
            return {}
        if not self._graph.text_nodes:
            return {}
        query_embs = self._embed_texts([query])
        top_clips, clip_ranked, _ = retrieve_from_graph(self._graph, query_embs, topk=topk, threshold=0.0)
        memories = collect_memories(self._graph, top_clips)
        clips = []
        for clip_id in top_clips:
            score = 0.0
            for cid, s in clip_ranked:
                if cid == clip_id:
                    score = s
                    break
            clips.append({
                "clip_id": clip_id,
                "score": score,
                "episodic": memories.get(clip_id, {}).get("episodic", []),
                "semantic": memories.get(clip_id, {}).get("semantic", []),
                "images": list(self._clip_images.get(clip_id, []))
            })
        return {"query": query, "clips": clips}

    def configure_generator(self, llm_engine, enabled: bool = True) -> None:
        self._generator_llm = llm_engine
        self._generator_enabled = enabled and llm_engine is not None

    def _add_multimodal_memory(
        self,
        belief: str,
        intention: str,
        state: str,
        verification: str,
        commands: Optional[List[Tuple[str, str]]],
        interaction_memory: Optional[str],
        image_paths: Any,
        raw_planner_output: Optional[str],
        task_context: Optional[str]
    ) -> None:
        clip_id = self._next_clip_id
        self._next_clip_id += 1

        episodic_text = None
        semantic_text = None
        if self._generator_enabled:
            episodic_text, semantic_text = self._generate_memory_via_llm(
                clip_id=clip_id,
                belief=belief,
                intention=intention,
                state=state,
                verification=verification,
                commands=commands,
                interaction_memory=interaction_memory,
                raw_planner_output=raw_planner_output,
                task_context=task_context,
                image_paths=image_paths
            )

        if not episodic_text:
            episodic_text = self._format_episodic_text(
                belief=belief,
                intention=intention,
                state=state,
                verification=verification,
                commands=commands,
                interaction_memory=interaction_memory,
                clip_id=clip_id
            )
        if not semantic_text:
            semantic_text = self._format_semantic_text(
                belief=belief,
                intention=intention,
                state=state,
                verification=verification,
                interaction_memory=interaction_memory,
                clip_id=clip_id
            )

        texts = [episodic_text]
        types = ["episodic"]
        if semantic_text:
            texts.append(semantic_text)
            types.append("semantic")

        text_embeddings = self._embed_texts(texts)
        text_node_ids = []
        for text, emb, mem_type in zip(texts, text_embeddings, types):
            node_id = self._graph.add_text_node(
                {"contents": [text], "embeddings": [emb]},
                clip_id=clip_id,
                text_type=mem_type
            )
            text_node_ids.append(node_id)

        if image_paths:
            from ...utils.utils import normalize_image_inputs
            normalized = normalize_image_inputs(image_paths)
            if normalized["paths"]:
                self._clip_images[clip_id] = list(normalized["paths"])
                img_embeddings = self._embed_images(normalized["paths"])
                if img_embeddings:
                    img_node_id = self._graph.add_img_node(
                        {"contents": list(normalized["paths"]), "embeddings": img_embeddings}
                    )
                    for text_node_id in text_node_ids:
                        self._graph.add_edge(img_node_id, text_node_id)

    def _format_episodic_text(
        self,
        belief: str,
        intention: str,
        state: str,
        verification: str,
        commands: Optional[List[Tuple[str, str]]],
        interaction_memory: Optional[str],
        clip_id: int
    ) -> str:
        parts = [
            f"CLIP_{clip_id} Episodic Memory",
            f"Belief: {belief}",
            f"Intention: {intention}",
            f"State: {state}",
            f"Verification: {verification}",
        ]
        if commands:
            cmd_text = "; ".join([f"{name}({params})" for name, params in commands])
            parts.append(f"Commands: {cmd_text}")
        if interaction_memory:
            parts.append(f"Interaction: {interaction_memory}")
        return "\n".join(parts)

    def _format_semantic_text(
        self,
        belief: str,
        intention: str,
        state: str,
        verification: str,
        interaction_memory: Optional[str],
        clip_id: int
    ) -> Optional[str]:
        summary_parts = []
        if belief:
            summary_parts.append(f"Belief summary: {belief}")
        if intention:
            summary_parts.append(f"Intention summary: {intention}")
        if state:
            summary_parts.append(f"State: {state}")
        if verification:
            summary_parts.append(f"Verifier feedback: {verification}")
        if interaction_memory:
            summary_parts.append(f"Interaction: {interaction_memory}")
        if not summary_parts:
            return None
        return f"CLIP_{clip_id} Semantic Memory\n" + "\n".join(summary_parts)

    def _generate_memory_via_llm(
        self,
        clip_id: int,
        belief: str,
        intention: str,
        state: str,
        verification: str,
        commands: Optional[List[Tuple[str, str]]],
        interaction_memory: Optional[str],
        raw_planner_output: Optional[str],
        task_context: Optional[str],
        image_paths: Any
    ) -> Tuple[Optional[str], Optional[str]]:
        if not self._generator_llm:
            return None, None

        commands_text = ""
        if commands:
            commands_text = "; ".join([f"{name}({params})" for name, params in commands])

        prompt = f"""
You are a memory generator for a VLN embodied agent.
Generate two memories for the current step:
1) episodic_memory: concrete observations, actions, and immediate outcomes.
2) semantic_memory: higher-level conclusion useful for long-horizon planning.

Return JSON only:
{{
  "episodic_memory": "...",
  "semantic_memory": "..."
}}

Context:
- Clip ID: {clip_id}
- Task: {task_context or "N/A"}
- Belief: {belief}
- Intention: {intention}
- State: {state}
- Commands: {commands_text or "None"}
- Verification: {verification}
- Interaction Memory: {interaction_memory or "None"}
- Planner Output (raw): {raw_planner_output or "None"}
"""

        input_data = [prompt]
        if image_paths:
            try:
                from ...utils.utils import append_image_bytes
                append_image_bytes(input_data, image_paths, log_prefix="Memory Generator")
            except Exception as e:
                print(f"[Memory] Failed to append image bytes: {e}")

        try:
            response = self._generator_llm(input_data)
        except Exception as e:
            print(f"[Memory] Memory generator failed: {e}")
            return None, None

        episodic = None
        semantic = None
        if isinstance(response, str):
            try:
                data = json.loads(response)
                episodic = data.get("episodic_memory")
                semantic = data.get("semantic_memory")
            except Exception:
                episodic = None
                semantic = None
        return episodic, semantic

    def _embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        texts = list(texts)
        if not texts:
            return []

        if self._embedding_backend == "local":
            try:
                if self._local_embedder is None:
                    if SentenceTransformer is None:
                        raise RuntimeError("sentence-transformers is not installed.")
                    model_name = os.environ.get("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
                    self._local_embedder = SentenceTransformer(model_name)
                embeddings = self._local_embedder.encode(texts, normalize_embeddings=True)
                return [emb.tolist() for emb in embeddings]
            except Exception as e:
                print(f"[Memory] Local embeddings failed, falling back to hash embeddings: {e}")
                return [self._hash_embedding(text) for text in texts]

        if openai is not None:
            try:
                if self._openai_client is None:
                    self._openai_client = openai.OpenAI()
                embeddings = self._openai_client.embeddings.create(
                    input=texts,
                    model=self._embedding_model
                )
                return [item.embedding for item in embeddings.data]
            except Exception as e:
                print(f"[Memory] OpenAI embeddings failed, falling back to hash embeddings: {e}")

        return [self._hash_embedding(text) for text in texts]

    def _embed_images(self, image_paths: List[str]) -> List[List[float]]:
        if not image_paths:
            return []
        if self._embedding_backend == "local":
            try:
                if self._local_image_embedder is None:
                    if SentenceTransformer is None:
                        raise RuntimeError("sentence-transformers is not installed.")
                    self._local_image_embedder = SentenceTransformer(self._image_embedding_model)
                if Image is None:
                    raise RuntimeError("PIL is not installed.")
                images = []
                for path in image_paths:
                    try:
                        images.append(Image.open(path).convert("RGB"))
                    except Exception:
                        continue
                if not images:
                    return []
                embeddings = self._local_image_embedder.encode(images, normalize_embeddings=True)
                return [emb.tolist() for emb in embeddings]
            except Exception as e:
                print(f"[Memory] Local image embeddings failed, falling back to hash embeddings: {e}")
        return [self._hash_embedding(path) for path in image_paths]

    @staticmethod
    def _hash_embedding(text: str, dim: int = 384) -> List[float]:
        vec = [0.0] * dim
        if not text:
            return vec
        data = text.encode("utf-8", errors="ignore")
        for i, b in enumerate(data):
            vec[i % dim] += (b % 13) - 6
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
        norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _mean_vector(vectors: List[List[float]]) -> List[float]:
        if not vectors:
            return []
        dim = len(vectors[0])
        summed = [0.0] * dim
        for vec in vectors:
            for i, x in enumerate(vec):
                summed[i] += x
        return [x / len(vectors) for x in summed]

    @staticmethod
    def _l2_distance(a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def compute_observation_uncertainty(self, image_paths: Any) -> float:
        """
        Umemory = ||f(o_t) - f(M_t)||: inconsistency between the current observation
        and the most recently stored image memory (M_t = previous step's img node).
        Returns 0.0 when disabled, no current image, or no prior memory to compare against.
        """
        if self.is_enable == False or not image_paths:
            return 0.0

        last_img_node = None
        for node_id in sorted(self._graph.nodes.keys(), reverse=True):
            node = self._graph.nodes[node_id]
            if node.type == "img":
                last_img_node = node
                break

        if last_img_node is None or not last_img_node.embeddings:
            return 0.0

        from ...utils.utils import normalize_image_inputs
        normalized = normalize_image_inputs(image_paths)
        if not normalized["paths"]:
            return 0.0

        obs_embeddings = self._embed_images(normalized["paths"])
        if not obs_embeddings:
            return 0.0

        obs_vec = self._mean_vector(obs_embeddings)
        mem_vec = self._mean_vector(last_img_node.embeddings)
        return self._l2_distance(obs_vec, mem_vec)

    def parse_vln_output(self, output_text: str) -> Dict[str, Any]:
        """
        Parse the VLN output text to extract Belief, Intention, and State sections.
        Returns a dictionary with parsed data for each section.
        """
        log_path = Path("tmp/llm_raw_text.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"Parsing VLN output: {output_text}\n" + "-"*80 + "\n")

        def extract_section(section_name: str, pattern: str):
            match = re.search(pattern, output_text, re.IGNORECASE | re.DOTALL)
            if match:
                section_text = match.group(1).strip()
                print(f"Extracted {section_name} Text: {section_text}")
                return section_text
            else:
                print(f"Label '{section_name}:' not found in output.")
                return None

        parsed_data = {}

        # Parse Belief
        belief_text = extract_section(
            "Belief",
            r"(?:\*\*Belief\*\*|Belief)\s*:\s*(.*?)(\n(?:Intention|State|Action|Description|$))"
        )
        if belief_text:
            belief_dict = {}
            lines = belief_text.splitlines()
            for line in lines:
                if line.strip().startswith('-'):
                    key_value = line.strip()[1:].strip().split(':', 1)
                    if len(key_value) == 2:
                        key = key_value[0].strip()
                        value = key_value[1].strip()
                        belief_dict[key] = value
            parsed_data['belief'] = belief_dict

        # Parse Intention
        intention_text = extract_section(
            "Intention",
            r"(?:\*\*Intention\*\*|Intention)\s*:\s*(.*?)(\n(?:State|Action|Belief|Description|$))"
        )
        if intention_text:
            reasoning_match = re.search(r"\[Next step reasoning\]:\s*(.*?)(?=\[Area of interest\]|$)", intention_text, re.DOTALL)
            area_match = re.search(r"\[Area of interest\]:\s*(.*)", intention_text, re.DOTALL)
            intention = {
                "reasoning": reasoning_match.group(1).strip() if reasoning_match else None,
                "area": area_match.group(1).strip() if area_match else None
            }
            parsed_data['intention'] = intention

        # Parse State
        state_text = extract_section(
            "State",
            r"(?:\*\*State\*\*|State)\s*:\s*(.*?)(\n(?:Action|Intention|Belief|Description|$))"
        )
        if state_text:
            state_match = re.search(r"<(.*?)>", state_text, re.DOTALL)
            if state_match:
                parsed_data['state'] = state_match.group(1).strip()

        return parsed_data

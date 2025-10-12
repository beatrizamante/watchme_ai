def save_state(self, stage, data):
        """Save pipeline state to resume later"""
        state = {
            "stage": stage,
            "data": data
        }
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"✓ Pipeline state saved: {stage}")

    def load_state(self):
        """Load previous pipeline state"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            print(f"✓ Resuming from stage: {state['stage']}")
            return state
        return None
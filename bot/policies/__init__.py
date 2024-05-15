import os
import importlib.util
from typing import Dict, Type
from policies.policy import Policy

def load_policies() -> Dict[str, Type[Policy]]:
    """
    Dynamically load policy classes from the 'policies' directory.

    :return: A dictionary mapping policy class names to their respective classes.
    """
    policy_classes = {}

    # Get the absolute path of the 'policies' directory
    policies_dir = os.path.dirname(os.path.abspath(__file__))

    # Iterate over the files in the 'policies' directory
    for file_name in os.listdir(policies_dir):
        if file_name.endswith(".py") and file_name != "__init__.py" and file_name != "policy.py":
            # Construct the module name and file path
            module_name = f"policies.{file_name[:-3]}"
            file_path = os.path.join(policies_dir, file_name)

            # Load the module using importlib.util
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Iterate over the attributes of the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)

                # Check if the attribute is a class and a subclass of Policy
                if isinstance(attr, type) and issubclass(attr, Policy) and attr != Policy:
                    policy_classes[attr_name] = attr

    return policy_classes

# Load the policy classes when the module is imported
policy_classes = load_policies()

print(policy_classes)
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import click
import time
import yaml
import random
import string
import os

@click.command()
@click.option("--pod_yaml", "-p", default="/home/eidf095/eidf095/crae-ml/smartback/my_pod.yaml", type=str)
def main(pod_yaml):
    os.system("cd /home/eidf095/eidf095/crae-ml/smartback/; sudo docker build -t=bigballoon8/custom-backprop:latest .; sudo docker push bigballoon8/custom-backprop")
    try:
        namespace = "eidf095ns"
        config.load_kube_config()
        v1 = client.CoreV1Api()

        with open(pod_yaml, "r") as stream:
            pod_def = yaml.safe_load(stream)
        
        base_identifier =''.join(random.choices(string.ascii_lowercase, k=8))
        pod_def["metadata"]["generateName"] = f"{pod_def['metadata']['generateName']}{base_identifier}-"
        base_pod_name = pod_def["metadata"]["generateName"]

        v1.create_namespaced_pod(body=pod_def, namespace=namespace)                

        pods = v1.list_namespaced_pod(namespace)
        pod_names = [pod.metadata.name for pod in pods.items]
        # Pod name should have rand string
        count = 1
        while True:
            for pod_name in pod_names:
                if base_pod_name in pod_name:
                    pod = pod_name
                    break
            else:
                # Allow 2 minutes 
                if count == 12:
                    raise ValueError(f"Pod:{base_pod_name} not found in available pods {pod_names}")
                click.echo(f"Pod:{base_pod_name} not found in available pods {pod_names}")
                time.sleep(10)
                count += 1
                continue
            break

        status = ""
        start_idx = 0
        while status not in ("Failed", "Succeeded"):
            status = v1.read_namespaced_pod(name=pod, namespace=namespace).status.phase
            try:
                pod_log = v1.read_namespaced_pod_log(name=pod_name, namespace=namespace)
                if pod_log[start_idx:]:
                    click.echo(pod_log[start_idx:])
                    start_idx = len(pod_log)
            except ApiException:
                pass
            time.sleep(0.5)

        time.sleep(4)

        try:
            pod_log = v1.read_namespaced_pod_log(name=pod_name, namespace=namespace)
            if pod_log[start_idx:]:
                click.echo(pod_log[start_idx:])
                start_idx = len(pod_log)
        except ApiException:
            pass

        pod_log = v1.read_namespaced_pod_log(name=pod_name, namespace=namespace)
        click.echo(pod_log[start_idx:])

        v1.delete_namespaced_pod(name=pod_name, namespace=namespace)
    except KeyboardInterrupt:
        while True:
            for pod_name in pod_names:
                if base_pod_name in pod_name:
                    pod = pod_name
                    print(f"Found Pod: {pod}")
                    break
            else:
                raise KeyboardInterrupt()
            break
        v1.delete_namespaced_pod(name=pod_name, namespace=namespace)
        raise KeyboardInterrupt()
    
if __name__ == "__main__":
    main()
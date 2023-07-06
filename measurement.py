import json
import re
import os
import sqlite3
import csv
import argparse

threshold = {
  'resnet20': 100*(1-0.0875),
  'resnet56': 100*(1-0.0697),
  'resnet110': 100*(1-0.0643)
}
sqlite_path = "/tmp/gpu-metric.sqlite"
# nvtx_list = ["prepare data", "forward", "gradient clean", "backpropagation", "gradient update"]
nvtx_list = []#["prepare data", "forward", "gradient clean", "backpropagation", "gradient update"]


def mean(list):
  sum = 0
  for item in list:
    sum += item
  return sum/len(list)


def Baseline(path):
  assert os.path.exists(path)

  # Read all the content in the log file
  with open(path, 'r') as logf:
    content = logf.read()

  # Get the arch of model and batch size
  arch = re.search('arch : (resnet[0-9]+)', content).group(1)
  batch_size = int(re.search('batch_size : ([0-9]+)', content).group(1))
  step_size = (50000 + batch_size - 1) // batch_size

  metric = {
    'Step Time(ms)': [],
    'Accuracy(%)': []
  }

  # Read the log file line by line
  for line in content.split('\n'):
    gst = re.search('Batch Time: ([0-9.]*)ms', line)
    gacc = re.search('Test: Accuracy: ([0-9.]*)%', line)
    gtt = re.search('Time used: ([0-9.]*)s', line)
    if gst:
      metric['Step Time(ms)'].append(float(gst.group(1)))
    if gacc:
      metric['Accuracy(%)'].append(float(gacc.group(1)))
    if gtt:
      if 'TTA(s)' not in metric and metric['Accuracy(%)'][-1]>=threshold[arch]:
        metric['TTA(s)'] = float(gtt.group(1))

  metric['Throughput(img/s)'] = [50000 * 1e3 / (st * step_size) for st in metric['Step Time(ms)']]
  # Exclude the first epoch
  metric['Step Time(ms)'] = mean(metric['Step Time(ms)'][1:])
  metric['Throughput(img/s)'] = mean(metric['Throughput(img/s)'][1:])
  metric['Accuracy(%)'] = mean(metric['Accuracy(%)'][-1:-6:-1])

  # Format
  metric = dict([(k, f'{v:.2f}') for k, v in metric.items()])
  # Add informations
  metric["Arch"] = arch
  metric["Batch Size"] = batch_size

  return metric


def GPUMetric(path):
  # Export qdrep to sqlite
  ret = os.system(f"nsys export -t sqlite -o {sqlite_path} {path}")
  assert ret == 0

  # Open the database
  conn = sqlite3.connect(sqlite_path)

  epochs = {}
  # Process NVTX events
  cursor = conn.execute("SELECT start, end, text from NVTX_EVENTS")

  for row in cursor:
    # Filter out "epoch"
    if "epoch" in row[2]:
      epoch = row[2]
      epochs[epoch] = {'timestamp': row[:2]}
      epochs[epoch].update(dict([(k, []) for k in nvtx_list]))

    if row[2] in nvtx_list:
      epochs[epoch][row[2]].append(row[:2])

  # Process generic events(GPU matrics)
  proc_metrics = ["GR Active", "Compute Warps In Flight", "Tensor Active",
                  "PCIe TX Throughput", "PCIe RX Throughput",
                  "NVLink TX Throughput", "NVLink RX Throughput"]
  cursor = conn.execute("SELECT rawTimestamp, data from GENERIC_EVENTS")
  # Status init
  lstTime = None
  epoch_id = 0
  metrics = []
  epoch_data = dict([(k, .0) for k in proc_metrics])
  phase_breakdown = dict([k, .0] for k in nvtx_list)
  phase_time = dict([k, .0] for k in nvtx_list)
  for row in cursor:
    s, e = epochs[f'epoch:{epoch_id}']['timestamp']
    if lstTime:
      start_in_nvtx = s < lstTime < e
      end_in_nvtx = s < row[0] < e
      if start_in_nvtx:
        # Accumlate metrics for epoch
        data = json.loads(row[1])
        for k in proc_metrics:
          epoch_data[k] += float(data[k]) / 100 * (row[0] - lstTime)

        # Accumlate metrics for phases
        for k in nvtx_list:
          if len(epochs[f'epoch:{epoch_id}'][k]) == 0:
            continue
          # Phase start, phase end
          ps, pe = epochs[f'epoch:{epoch_id}'][k][0]
          if ps < lstTime < pe:
            phase_breakdown[k] += float(data["GR Active"]) / 100 * (row[0] - lstTime)
            # Leave off the phase
            if not ps < row[0] < pe:
              phase_time[k] += pe - ps
              epochs[f'epoch:{epoch_id}'][k] = epochs[f'epoch:{epoch_id}'][k][1:]
            break

        # Leave off the epoch
        if not end_in_nvtx:
          # Save phases time
          epochs[f'epoch:{epoch_id}'].update(phase_time)
          epoch_data.update(phase_breakdown)
          # Calc GCT
          epoch_data['GPU Computing Time(s)'] = epoch_data['GR Active']
          metrics.append(epoch_data)
          epoch_id += 1
          if f'epoch:{epoch_id}' not in epochs:
            break
          # Reset status
          epoch_data = dict([(k, .0) for k in proc_metrics])
          phase_breakdown = dict([k, .0] for k in nvtx_list)
          phase_time = dict([k, .0] for k in nvtx_list)
    
    lstTime = row[0]

  # remove sqlite file
  ret = os.system(f"rm {sqlite_path}")
  assert ret == 0

  # Exclude the first epoch
  metrics = metrics[1:]
  del epochs['epoch:0']
  # Calc the avg of metrics(epoch)
  avg_metric = {}
  avg_metric['GPU Computing Time(s)'] = f"{mean([m['GPU Computing Time(s)'] for m in metrics]) * 1e-9:>.2f}"
  epoch_time = mean([epochs[k]['timestamp'][1] - epochs[k]['timestamp'][0] for k in epochs.keys()])
  for key in proc_metrics:
    avg_metric[key] = f"{mean([m[key] for m in metrics]) * 100 / epoch_time:.2f}"

  # Calc the avg of metrics(phase)
  for key in nvtx_list:
    phase_time = mean([epochs[k][key] for k in epochs.keys()])
    avg_metric[key+'(ms)'] = f"{phase_time * 1e-6:>.2f}"
    if phase_time != 0:
      avg_metric[key+'(%)'] = f"{mean([m[key] for m in metrics]) * 100 / phase_time:.2f}"
    else:
      avg_metric[key+'(%)'] = "0"

  return avg_metric


def parse_arg():
  parser = argparse.ArgumentParser()
  parser.add_argument('--root', default="", type=str)
  parser.add_argument('--device', default="a6000", type=str)
  args = parser.parse_args()
  return args

header = ["Arch", "Batch Size", "Step Time(ms)", "Accuracy(%)", "TTA(s)", "Throughput(img/s)",
          "PCIe TX Throughput", "PCIe RX Throughput", "NVLink TX Throughput",
          "NVLink RX Throughput", "GR Active", "Tensor Active", "Compute Warps In Flight",
          "GPU Computing Time(s)"]
header += [x + "(ms)" for x in nvtx_list]
header += [x + "(%)" for x in nvtx_list]


if __name__ == '__main__':
  args = parse_arg()

  metrics = []
  path = os.path.join(args.root, "experiments")
  assert os.path.exists(path)

  # All the dir
  dirs = os.listdir(path)

  # Baseline if exists
  if "baseline" in dirs:
    base_path = os.path.join(path, "baseline")
    for fname in os.listdir(base_path):
      print(f"processing log file {fname} ...")
      metrics.append(Baseline(os.path.join(base_path, fname)))
    dirs.remove("baseline")

  # GPU metrics
  for dir in dirs:
    g = re.match('([a-zA-Z0-9-]*)-([0-9]*)', dir)
    # Assert the dirname is correct
    assert g
    metric = {"Arch": g.group(1), "Batch Size": int(g.group(2))}
    # Remove the exist item in metrics
    for m in metrics:
      if m["Arch"] == metric["Arch"] and m["Batch Size"] == metric["Batch Size"]:
        metric.update(m)
        metrics.remove(m)
        break
    # Assert .qdrep exists
    qdrep = os.path.join(os.path.join(path, dir), f"{args.device}.qdrep")
    if os.path.isfile(qdrep):
      print(f"processing .qdrep in {dir}/ ...")
      metric.update(GPUMetric(qdrep))
      metrics.append(metric)

  def sortkey(e):
    arch = int(re.search('([0-9]+)', e["Arch"]).group(1))
    bs = e["Batch Size"]
    return (arch, bs)
  metrics.sort(key=sortkey)

  with open(os.path.join(args.root, "metrics.csv"), "w") as out_f:
    csv_f = csv.DictWriter(out_f, header)
    csv_f.writeheader()
    for metric in metrics:
      csv_f.writerow(metric)

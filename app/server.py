from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
from fastai.vision.all import *

model_file_url = 'https://drive.google.com/uc?export=download&id=1gsFNi__Yu7KbFHqiRpzEd4_udak8HD0O'
model_file_name = 'export.pkl'
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

class RelativeSelfAttention(Module):
    def __init__(self, d_in, d_out, ks, groups, stride=1):
        self.n_c, self.ks, self.groups, self.stride = d_out, ks, groups, stride
        # linear transformation for queries, values and keys
        self.qx, self.kx, self.vx = [ConvLayer(d_in, d_out, ks=1, norm_type=None,
                                               act_cls=None) for _ in range(3)]
        # positional embeddings
        self.row_embeddings = nn.Parameter(torch.randn(d_out//2, ks))
        self.col_embeddings = nn.Parameter(torch.randn(d_out//2, ks))
        
    def calc_out_shape(self, inp_shape, pad):
        out_shape = [(sz + 2*pad - self.ks) // self.stride + 1 for sz in inp_shape]
        return out_shape
    
    def forward(self, x):
        query, keys, values = self.qx(x), self.kx(x), self.vx(x)
        
        pad = (self.ks -1) // 2
        
        # use unfold to extract the memory blocks and their associated queries
        query = F.unfold(query, kernel_size=1, stride=self.stride)
        keys = F.unfold(keys, kernel_size=self.ks, padding=pad, stride=self.stride)
        values = F.unfold(values, kernel_size=self.ks, padding=pad, stride=self.stride)
        
        
        # reshape and permute the dimensions into the appropriate format for matrix multiplication
        query = query.view(query.shape[0], self.groups, self.n_c//self.groups, -1, query.shape[-1]) # bs*G*C//G*1*N
        query = query.permute(0, 4, 1, 2, 3) # bs * N * G * C//G * 1
        keys = keys.view(keys.shape[0], self.groups, self.n_c//self.groups, -1, keys.shape[-1]) # bs*G*C//G*ks^2*N
        keys = keys.permute(0, 4, 1, 2, 3) # bs * N * G * C//G * ks^2
        values = values.view(values.shape[0], self.groups, self.n_c//self.groups, -1, values.shape[-1]) # bs*G*C//G*ks^2*N
        values = values.permute(0, 4, 1, 2, 3) # bs * N * G * C//G * ks^2
        
        # get positional embeddings
        row_embeddings = self.row_embeddings.unsqueeze(-1).expand(-1, -1, self.ks)
        col_embeddings = self.col_embeddings.unsqueeze(-2).expand(-1, self.ks, -1)
        
        embeddings = torch.cat((row_embeddings, col_embeddings)).view(self.groups,
                                self.n_c//self.groups, -1) # G * C//G * ks^2
        # add empty dimensions to match the shape of keys
        embeddings = embeddings[None, None, -1] # 1 * 1 * G * C//G * ks^2
        
        # compute attention map
        att_map = F.softmax(torch.matmul(query.transpose(-2,-1), keys+embeddings).contiguous(), dim=-1)
        # compute final output
        out = torch.matmul(att_map, values.transpose(-2,-1)).contiguous().permute(0, 2, 3, 4, 1)
        
        return out.view(out.shape[0], self.n_c, *self.calc_out_shape(x.shape[-2:], pad)).contiguous()
        
def resnet_stem(*sizes):
    return [
        ConvLayer(sizes[i], sizes[i+1], stride=2 if i==0 else 1)
         for i in range(len(sizes) - 1)
    ] + [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
    
def bottleneck(ni, nf, stride):
    if stride==1:
        layers = [ConvLayer(ni, nf//4, ks=1),
              RelativeSelfAttention(nf//4, nf//4, ks=7, groups=8),
              ConvLayer(nf//4, nf, ks=1, act_cls=None, norm_type=NormType.BatchZero)]
    else:
        layers = [ConvLayer(ni, nf//4, ks=1),
              RelativeSelfAttention(nf//4, nf//4, ks=7, groups=8),
              nn.AvgPool2d(2, ceil_mode=True),
              ConvLayer(nf//4, nf, ks=1, act_cls=None, norm_type=NormType.BatchZero)]
    
    return nn.Sequential(*layers)
    
class ResNetBlock(Module):
    def __init__(self, ni, nf, stride, sa, expansion=1):
        self.botl = bottleneck(ni, nf, stride)
        self.idconv = noop if ni==nf else ConvLayer(ni, nf, 1, act_cls=None)
        self.pool = noop if stride==1 else nn.AvgPool2d(2, ceil_mode=True)
        
    def forward(self, x):
        return F.relu(self.botl(x) + self.idconv(self.pool(x)))
        
class RandHead(Module):
    def __init__(self):
        pass
        
    def forward(self, x):
        rz_shape = torch.randint(210, 224, (1,))
        pad = (224 - rz_shape).item()
        h = torch.randint(0, pad+1, (1,))
        w = torch.randint(0, pad+1, (1,))
        
        # step 1 random resize
        out = F.interpolate(x, [rz_shape]*2)
        # step 2 pad
        return F.pad(out, (h, pad-h, w, pad-w))
        
class xResNet(nn.Sequential):
    def __init__(self, channels, n_out, blocks, sa=True, expansion=1):
        stem = resnet_stem(channels, 32, 32, 64)
        self.group_sizes = [64, 64, 128, 256, 512]
        for i in range(1, len(self.group_sizes)): 
            self.group_sizes[i] *= expansion
        groups = [self._make_group(idx, n_blocks, sa=sa if idx==0 else False) 
                      for idx, n_blocks in enumerate(blocks)]
        
        super().__init__(RandHead(), *stem, *groups,
                         nn.AdaptiveAvgPool2d(1), Flatten(),
                         nn.Linear(self.group_sizes[-1], n_out))
        
    def _make_group(self, idx, n_blocks, sa):
        stride = 1 if idx==1 else 2
        ni, nf = self.group_sizes[idx], self.group_sizes[idx+1]
        return nn.Sequential(*[
            ResNetBlock(ni if i==0 else nf, nf, stride=stride if i==0 else 1,
                        sa=sa if i==n_blocks-1 else False)
             for i in range(n_blocks)
        ])
        
def get_y(o):
    return [o.parent.name]

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(model_file_url, path/'models'/f'{model_file_name}')
    try:
        learn = load_learner(path/'models'/model_file_name, cpu=True)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

PREDICTION_FILE_SRC = path/'static'/'predictions.txt'

@app.route("/upload", methods=["POST"])
async def upload(request):
    form = await request.form()
    img_bytes = await (form["file"].read())
    return predict_from_bytes(img_bytes)

def predict_from_bytes(img_bytes):
    pred,pred_idx,probs = learn.predict(img_bytes)
    #classes = learn.dls.vocab
    #predictions = sorted(zip(classes, map(float, probs)), key=lambda p: p[1], reverse=True)
    
    prob = probs[pred_idx]
    
    result_html1 = path/'static'/'result1.html'
    result_html2 = path/'static'/'result2.html'
    

    if prob < 0.95:
    	result_html = str(result_html1.open().read() + "Input image is likely out of domain. Please upload a blood smear image" + result_html2.open().read())
    else:
    	prob = round(prob.item()*100, 2)
    	result_html = str(result_html1.open().read() + "Image is <strong>" + str(pred[0]) + "</strong>. Probability <strong>" + str(prob) + "%</strong>" + result_html2.open().read())
    
    # result_html = str(result_html1.open().read() +str(predictions[0:2]) + result_html2.open().read())
    return HTMLResponse(result_html)

@app.route("/")
def form(request):
    index_html = path/'static'/'index.html'
    return HTMLResponse(index_html.open().read())

if __name__ == "__main__":
    if "serve" in sys.argv: uvicorn.run(app, host="0.0.0.0", port=8080)

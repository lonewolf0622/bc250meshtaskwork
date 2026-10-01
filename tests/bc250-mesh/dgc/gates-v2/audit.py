"""Pre-submit pointer audit of MONO PM4 and the captured DGC template."""
import hashlib,re,struct
from pointer_check import check, fields, MASK48

def audit(text, side, case):
    assert 'BC250PREIB_END' in text, 'missing pre-submit CPU IB capture'
    if side == 'mono':
        result=check(text)
        result['before_submit']=True
        return result
    data=(case/'capture-0.bin').read_bytes()
    params=struct.unpack_from('<QQQQIIII',data)
    output,code=params[1],params[6]
    snapshot=data[48:]
    assert code==1048576 and len(snapshot)==params[5]
    # The private DGC upload is a CPU snapshot aliased at the final preprocess
    # arena VA. Preprocessing copies this data unchanged; application root
    # constants are patched from the stream into their dedicated block.
    text+='\nBC250MEM va=%x bytes=%d data=%s\n'%(output+code,len(snapshot)-code,snapshot[code:].hex())
    memory=[]
    for line in text.splitlines():
        if line.startswith('BC250MEM '):
            f=fields(line);b=bytes.fromhex(f['data'])
            assert len(b)==int(f['bytes'])
            memory.append((int(f['va'],16)&MASK48,b))
    def read(va,n):
        va &= MASK48
        assert va,'null pointer'
        for base,b in memory:
            if base<=va and n<=len(b)-(va-base):return b[va-base:va-base+n]
        raise AssertionError('unresolved pointer %x+%d'%(va,n))
    records=[fields(l) for l in text.splitlines() if l.startswith('BC250PTR ')]
    descriptors=[fields(l) for l in text.splitlines() if l.startswith('BC250DESC ')]
    live=[]
    for f in records:
        if f.get('compat')!='1':continue
        words=[int(f['words'][i:i+8],16) for i in range(0,len(f['words']),8)]
        # State restoration after preparing the template emits application
        # words back into the ordinary primary. Those SGPRs are overwritten by
        # the captured template before its first actual Mesh/Fragment dispatch.
        # Only the exact restored application block is exempt, never a null
        # effective pointer inside a compatibility block in the template.
        if f['stage'] in ('4','7') and words[0]==7 and not any(words[1:]):
            assert not int(f['pc'],16) and not int(f['inline_source'],16)
            continue
        live.append(f)
    assert {int(f['stage']) for f in live}=={4,5,7}
    # A template contains one indirect setup and 1024 bounded Task/consumer
    # pairs. Validate every emitted constant source, not only the first pair.
    assert sum(f['stage']=='7' for f in live)==1024
    resolved=0
    setups=[f for f in records if f['stage']=='5' and f['mask']=='ff20']
    assert len(setups)==1
    for f in live+setups:
        mask=int(f['mask'],16);source=int(f['inline_source'],16)
        values=[int(f['words'][i:i+8],16) for i in range(0,len(f['words']),8)]
        effective={w:struct.unpack('<I',read(source+w*4,4))[0] if source else values[w]
                   for w in range(64) if mask & (1<<w)}
        for word in (0,2,6,8,10,12,14):
            if mask & (3<<word)==3<<word:
                va=effective[word] | effective[word+1]<<32
                assert va & MASK48, 'null private pointer stage='+f['stage']
                read(va,4)
                if word==6:assert struct.unpack('<I',read(va,4))[0]==7
                resolved+=1
        pc=int(f['pc'],16)
        if int(f['pc_reg']):read(pc,72)
        assert f['stage'] in ('4','5','7')
    descriptor_records=0
    for f in descriptors:
        if int(f['enabled'],16)!=3:continue
        table=int(f['table'],16)
        if int(f['indirect_reg']):read(table,8)
        for i,value in ((0,11),(1,13)):
            va=int(f['set'+str(i)],16);desc=struct.unpack('<IIII',read(va,16))
            resource=desc[0] | (desc[1]&0xffff)<<32
            assert desc[2]>=4 and struct.unpack('<I',read(resource,4))[0]==value
            if int(f['indirect_reg']):assert struct.unpack('<I',read(table+i*4,4))[0]==va&0xffffffff
            descriptor_records+=1
    assert descriptor_records>=6
    # Validate template SH constants against their actual memory or inline
    # sources. No GPU-written indirect counts or output records are invented.
    raw=struct.unpack_from('<'+'I'*(code//4),snapshot)
    regs={};matches=0
    def registers(op,b):
        if op in (0x76,0x9b):
            for j,v in enumerate(b[1:]):regs[0xb000+(b[0]&0xffff)*4+j*4]=v
        elif op==0x63:
            assert len(b)==4 and not b[0]&3 and not b[2]>>16
            for j in range(b[3]):regs[0xb000+b[2]*4+j*4]=struct.unpack('<I',read((b[0]|b[1]<<32)+j*4,4))[0]
    # The template inherits graphics shader-program registers from its owning
    # primary. Seed exactly the state at the call into the preprocess arena.
    owner_words=[]
    for pre in re.findall(r'BC250PREIB_BEGIN[^\n]*\n(.*?)BC250PREIB_END',text,re.S):
        owner_words += [int(w,16) for w in re.findall(r'^([0-9a-f]{8}) raw',pre,re.M)]
    i=0;called=False
    while i<len(owner_words):
        h=owner_words[i];n=1 if h==0xffff1000 or h>>30==2 else ((h>>16)&0x3fff)+2
        assert h>>30 in (2,3) and i+n<=len(owner_words)
        b=owner_words[i+1:i+n];op=(h>>8)&255
        if op==0x3f and ((b[0]|b[1]<<32)&MASK48)==(output&MASK48):
            called=True;break
        registers(op,b);i+=n
    assert called,'missing owning IB call'
    i=0
    by_stage={s:[f for f in live if int(f['stage'])==s] for s in (4,5,7)}
    seen={s:0 for s in (4,5,7)}
    shader_va={s:int(by_stage[s][0]['shader'],16) for s in (4,5,7)}
    def compare(stage):
        nonlocal matches
        pgm={4:0xb020,5:0xb830,7:0xb320}[stage]
        if regs.get(pgm)!=(shader_va[stage]>>8)&0xffffffff:return
        f=by_stage[stage][seen[stage]];seen[stage]+=1
        mask=int(f['mask'],16);source=int(f['inline_source'],16);reg=int(f['inline_reg']);slot=0
        words=[int(f['words'][j:j+8],16) for j in range(0,len(f['words']),8)]
        for w in range(64):
            if mask&(1<<w):
                value=struct.unpack('<I',read(source+w*4,4))[0] if source else words[w]
                assert regs[reg+slot*4]==value,'template constant source mismatch'
                slot+=1
        matches+=1
    while i<len(raw):
        h=raw[i];n=1 if h==0xffff1000 or h>>30==2 else ((h>>16)&0x3fff)+2
        assert h>>30 in (2,3) and i+n<=len(raw)
        b=raw[i+1:i+n];op=(h>>8)&255
        assert op not in (0x4d,0xaa,0xad)
        registers(op,b)
        if op in (0x15,0x16):compare(5)
        elif op==0x4c:compare(7);compare(4)
        i+=n
    assert seen=={s:len(by_stage[s]) for s in by_stage}, 'missing template pointer emission'
    return {'before_submit':True,'private_pointer_checks':resolved,'descriptor_checks':descriptor_records,
            'template_shader_dispatches':matches,'gpu_executed':False,
            'template_sha256':hashlib.sha256(snapshot[:code]).hexdigest()}
